from pathlib import Path
from time import time
from typing import Literal

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from config import create_logger
from data_processing import SRDataset
from model import SRResNet
from utils import (
    Metrics,
    format_time,
    load_checkpoint,
    plot_training_metrics,
    rgb_to_ycbcr,
    save_checkpoint,
)

SCALING_FACTOR: Literal[2, 4, 8] = 4
CROP_SIZE = 128

N_CHANNELS = 96
N_RES_BLOCKS = 16
LARGE_KERNEL_SIZE = 9
SMALL_KERNEL_SIZE = 3

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-3
EPOCHS = 100
LOAD_MODEL = False
DEV_MODE = False

NUM_WORKERS = 8

TRAIN_DATASET_PATH = Path("data/COCO2017_train")
VAL_DATASET_PATH = Path("data/COCO2017_test")

CHECKPOINTS_DIR = Path("checkpoints")
MODEL_NAME = "srresnet"
MODEL_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_model.safetensors"
STATE_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_state.pth"

logger = create_logger(log_level="INFO")


def train_step(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None,
    scheduler: OneCycleLR | None,
    device: Literal["cpu", "cuda"] = "cpu",
) -> tuple[float, list[float]]:
    total_loss = 0
    learning_rates = []

    model.train()

    for i, (hr_image_tensor, lr_image_tensor) in enumerate(data_loader):
        hr_image_tensor = hr_image_tensor.to(device, non_blocking=True)
        lr_image_tensor = lr_image_tensor.to(device, non_blocking=True)

        if scaler:
            with autocast(device):
                sr_image_tensor = model(lr_image_tensor)
                loss = loss_fn(sr_image_tensor, hr_image_tensor)
        else:
            sr_image_tensor = model(lr_image_tensor)
            loss = loss_fn(sr_image_tensor, hr_image_tensor)

        total_loss += loss.item()

        optimizer.zero_grad()

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])

        if i % 500 == 0:
            logger.debug(f"Processing batch {i}/{len(data_loader)}...")

    total_loss /= len(data_loader)

    return total_loss, learning_rates


def validation_step(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    device: Literal["cpu", "cuda"] = "cpu",
) -> tuple[float, float, float]:
    total_loss = 0
    total_psnr = 0
    total_ssim = 0

    model.eval()

    with torch.inference_mode():
        for hr_image_tensor, lr_image_tensor in data_loader:
            hr_image_tensor = hr_image_tensor.to(device, non_blocking=True)
            lr_image_tensor = lr_image_tensor.to(device, non_blocking=True)

            sr_image_tensor = model(lr_image_tensor)
            loss = loss_fn(sr_image_tensor, hr_image_tensor)

            y_hr_tensor = rgb_to_ycbcr(hr_image_tensor)
            y_sr_tensor = rgb_to_ycbcr(sr_image_tensor)

            psnr = psnr_metric(y_sr_tensor, y_hr_tensor)
            ssim = ssim_metric(y_sr_tensor, y_hr_tensor)

            total_loss += loss.item()
            total_psnr += psnr.item()
            total_ssim += ssim.item()

        total_loss /= len(data_loader)
        total_psnr /= len(data_loader)
        total_ssim /= len(data_loader)

    return total_loss, total_psnr, total_ssim


def train(
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    start_epoch: int,
    epochs: int,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    scaler: GradScaler | None = None,
    scheduler: OneCycleLR | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
) -> None:
    best_psnr = 0.0
    metrics = Metrics()
    metrics.epochs = epochs - start_epoch + 1

    try:
        for epoch in range(start_epoch, epochs + 1):
            start_time = time()

            train_loss, learning_rates = train_step(
                train_data_loader, model, loss_fn, optimizer, scaler, scheduler, device
            )

            val_loss, val_psnr, val_ssim = validation_step(
                val_data_loader, model, loss_fn, psnr_metric, ssim_metric, device
            )

            end_time = time() - start_time
            epoch_time = format_time(end_time)
            remaining_time = format_time(end_time * (epochs - epoch))

            current_lr = optimizer.param_groups[0]["lr"]

            metrics.learning_rates.extend(learning_rates)
            metrics.train_losses.append(train_loss)
            metrics.val_losses.append(val_loss)
            metrics.psnrs.append(val_psnr)
            metrics.ssims.append(val_ssim)

            logger.info(
                f"Epoch: {epoch}/{epochs} ({epoch_time}/{remaining_time}) | LR: {current_lr:.2e} | T loss: {train_loss:.4f} | V loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.2f}"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    MODEL_CHECKPOINT_PATH,
                    STATE_CHECKPOINT_PATH,
                    epoch,
                    model,
                    optimizer,
                    scaler,
                )

        plot_training_metrics(metrics)

    except KeyboardInterrupt:
        save_checkpoint(
            MODEL_CHECKPOINT_PATH,
            STATE_CHECKPOINT_PATH,
            epoch,
            model,
            optimizer,
            scaler,
        )
        exit(0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SRDataset(
        data_folder=TRAIN_DATASET_PATH,
        scaling_factor=SCALING_FACTOR,
        crop_size=CROP_SIZE,
        dev_mode=DEV_MODE,
    )

    val_dataset = SRDataset(
        data_folder=VAL_DATASET_PATH,
        scaling_factor=SCALING_FACTOR,
        crop_size=CROP_SIZE,
        dev_mode=DEV_MODE,
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True if device == "cuda" else False,
        num_workers=NUM_WORKERS,
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True if device == "cuda" else False,
        num_workers=NUM_WORKERS,
    )

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

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device) if device == "cuda" else None
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=MAX_LEARNING_RATE,
        total_steps=EPOCHS * len(train_data_loader),
        div_factor=MAX_LEARNING_RATE / LEARNING_RATE,
        final_div_factor=100,
    )

    if LOAD_MODEL:
        start_epoch = load_checkpoint(
            MODEL_CHECKPOINT_PATH,
            STATE_CHECKPOINT_PATH,
            model,
            optimizer,
            scaler,
            scheduler,
            device,
        )
    else:
        start_epoch = 1

    train(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        start_epoch=start_epoch,
        epochs=EPOCHS,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric,
        scaler=scaler,
        scheduler=scheduler,
        device=device,
    )


if __name__ == "__main__":
    main()
