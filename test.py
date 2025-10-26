from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from config import create_logger
from data_processing import SRDataset
from model import SRResNet
from utils import inverse_tta_transform, load_checkpoint, rgb_to_ycbcr, tta_transforms

SCALING_FACTOR: Literal[2, 4, 8] = 4

N_CHANNELS = 96
N_RES_BLOCKS = 16
LARGE_KERNEL_SIZE = 9
SMALL_KERNEL_SIZE = 3

BATCH_SIZE = 1

NUM_WORKERS = 8

CHECKPOINTS_DIR = Path("checkpoints")
MODEL_NAME = "srresnet"
MODEL_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_model_best.safetensors"
STATE_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_state_best.pth"

DATASETS_DIR = Path("data")
DATASETS = ["Set5", "Set14", "BSDS100", "Urban100"]

logger = create_logger(log_level="INFO")


def test_step(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    use_tta: bool = True,
    device: Literal["cpu", "cuda"] = "cpu",
) -> tuple[float, float, float]:
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = len(data_loader)

    model.eval()

    with torch.inference_mode():
        for hr_image_tensor, lr_image_tensor in data_loader:
            hr_image_tensor = hr_image_tensor.to(device, non_blocking=True)
            lr_image_tensor = lr_image_tensor.to(device, non_blocking=True)

            if use_tta:
                sr_images = []

                for tta_transform in tta_transforms:
                    sr_image_tensor = model(tta_transform(lr_image_tensor))
                    sr_image_tensor = inverse_tta_transform(
                        sr_image_tensor, tta_transform
                    )
                    sr_images.append(sr_image_tensor)

                sr_image_tensor = torch.mean(torch.stack(sr_images), dim=0)
            else:
                sr_image_tensor = model(lr_image_tensor)

            loss = loss_fn(sr_image_tensor, hr_image_tensor)

            y_hr_tensor = rgb_to_ycbcr(hr_image_tensor)
            y_sr_tensor = rgb_to_ycbcr(sr_image_tensor)

            sf = SCALING_FACTOR
            y_hr_tensor = y_hr_tensor[:, :, sf:-sf, sf:-sf]
            y_sr_tensor = y_sr_tensor[:, :, sf:-sf, sf:-sf]

            psnr = psnr_metric(y_sr_tensor, y_hr_tensor)
            ssim = ssim_metric(y_sr_tensor, y_hr_tensor)

            total_loss += loss.item()
            total_psnr += psnr.item()
            total_ssim += ssim.item()

    return total_loss / n_batches, total_psnr / n_batches, total_ssim / n_batches


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SRResNet(
        n_channels=N_CHANNELS,
        large_kernel_size=LARGE_KERNEL_SIZE,
        small_kernel_size=SMALL_KERNEL_SIZE,
        n_res_blocks=N_RES_BLOCKS,
        scaling_factor=SCALING_FACTOR,
    ).to(device)

    loss_fn = nn.MSELoss()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    optimizer = optim.Adam(model.parameters())

    _ = load_checkpoint(
        model_filepath=MODEL_CHECKPOINT_PATH,
        state_filepath=STATE_CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    for dataset_name in DATASETS:
        dataset = SRDataset(
            data_folder=DATASETS_DIR / dataset_name,
            scaling_factor=SCALING_FACTOR,
            test_mode=True,
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True if device == "cuda" else False,
            num_workers=NUM_WORKERS,
        )

        avg_loss, avg_psnr, avg_ssim = test_step(
            data_loader=data_loader,
            model=model,
            loss_fn=loss_fn,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric,
            use_tta=True,
            device=device,
        )

        logger.info(
            f"{dataset_name} Dataset | Avg loss: {avg_loss:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}"
        )


if __name__ == "__main__":
    main()
