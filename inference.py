from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from torch import nn
from torchvision.io import decode_image

from config import create_logger
from model import SRResNet
from utils import compare_images, inverse_tta_transform, load_checkpoint, tta_transforms

INPUT_PATH = Path("images/inference_img_5.jpg")
OUTPUT_PATH = Path("images/sr_img_5.png")
COMPARISON_IMAGE_PATH = Path("images/comparison_img_5.png")

SCALING_FACTOR: Literal[2, 4, 8] = 4
N_CHANNELS = 96
N_RES_BLOCKS = 16
LARGE_KERNEL_SIZE = 9
SMALL_KERNEL_SIZE = 3

CHECKPOINTS_DIR = Path("checkpoints")
MODEL_NAME = "srresnet"
MODEL_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_model.safetensors"
STATE_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_state.pth"

logger = create_logger(log_level="INFO")


def upscale_tiled(
    model: nn.Module,
    lr_tensor: torch.Tensor,
    scaling_factor: int,
    tile_size: int = 512,
    overlap: int = 64,
    device: Literal["cpu", "cuda"] = "cpu",
) -> torch.Tensor:
    if overlap >= tile_size:
        raise ValueError("Overlap must be smaller than tile_size")

    lr_b, lr_c, lr_h, lr_w = lr_tensor.shape

    border_pad = overlap // 2
    lr_tensor = F.pad(
        lr_tensor, (border_pad, border_pad, border_pad, border_pad), "reflect"
    )

    orig_hr_h = lr_h * scaling_factor
    orig_hr_w = lr_w * scaling_factor

    _, _, lr_h, lr_w = lr_tensor.shape

    step = tile_size - overlap

    pad_h = step - (lr_h - tile_size) % step
    if pad_h == step:
        pad_h = 0

    pad_w = step - (lr_w - tile_size) % step
    if pad_w == step:
        pad_w = 0

    lr_tensor_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), "reflect")
    _, _, padded_h, padded_w = lr_tensor_padded.shape

    padded_hr_h = padded_h * scaling_factor
    padded_hr_w = padded_w * scaling_factor

    hr_canvas = torch.zeros(
        lr_b, lr_c, padded_hr_h, padded_hr_w, dtype=torch.float32
    ).to("cpu")
    vote_canvas = torch.zeros_like(hr_canvas).to("cpu")

    num_tiles_h = (padded_h - overlap) // step
    num_tiles_w = (padded_w - overlap) // step

    logger.info(
        f"Starting tiled inference with overlap: {num_tiles_w}x{num_tiles_h} tiles..."
    )

    for h_idx in range(num_tiles_h):
        for w_idx in range(num_tiles_w):
            h_start = h_idx * step
            w_start = w_idx * step

            h_end = h_start + tile_size
            w_end = w_start + tile_size

            lr_tile = lr_tensor_padded[:, :, h_start:h_end, w_start:w_end]

            with torch.inference_mode():
                hr_tile = model(lr_tile.to(device))

            hr_h_start = h_start * scaling_factor
            hr_w_start = w_start * scaling_factor
            hr_h_end = h_end * scaling_factor
            hr_w_end = w_end * scaling_factor

            hr_canvas[:, :, hr_h_start:hr_h_end, hr_w_start:hr_w_end] += hr_tile.to(
                "cpu"
            )
            vote_canvas[:, :, hr_h_start:hr_h_end, hr_w_start:hr_w_end] += 1

    small_value = 1e-6
    hr_output_padded = hr_canvas / (vote_canvas + small_value)

    hr_border_pad = border_pad * scaling_factor
    return hr_output_padded[
        :,
        :,
        hr_border_pad : hr_border_pad + orig_hr_h,
        hr_border_pad : hr_border_pad + orig_hr_w,
    ]


def upscale_image(
    input_path: Path,
    output_path: Path,
    comparison_image_path: Path | None,
    scaling_factor: Literal[2, 4, 8] = 4,
    use_tta: bool = True,
    use_tiling: bool = True,
    use_downscale: bool = False,
    device: Literal["cpu", "cuda"] = "cpu",
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if input_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("Input image must be in JPG or PNG format")

    img_tensor_uint8 = decode_image(input_path.__fspath__())

    if img_tensor_uint8.shape[0] > 3:
        img_tensor_uint8 = img_tensor_uint8[:3]

    if use_downscale:
        logger.info(f"Downscaling image by {scaling_factor} times...")
        hr_c, hr_h, hr_w = img_tensor_uint8.shape

        height_remainder = hr_h % scaling_factor
        width_remainder = hr_w % scaling_factor

        if height_remainder != 0 or width_remainder != 0:
            top = height_remainder // 2
            left = width_remainder // 2
            bottom = top + (hr_h - height_remainder)
            right = left + (hr_w - width_remainder)
            hr_img_tensor_uint8 = img_tensor_uint8[:, top:bottom, left:right]
        else:
            hr_img_tensor_uint8 = img_tensor_uint8

        _, hr_h_cropped, hr_w_cropped = hr_img_tensor_uint8.shape
        lr_h = hr_h_cropped // scaling_factor
        lr_w = hr_w_cropped // scaling_factor

        lr_transform = transforms.Resize(
            size=(lr_h, lr_w),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        lr_img_tensor_uint8 = lr_transform(hr_img_tensor_uint8)

    else:
        lr_img_tensor_uint8 = img_tensor_uint8

    transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img_tensor = transform(lr_img_tensor_uint8).unsqueeze(0)

    model = SRResNet(
        n_channels=N_CHANNELS,
        large_kernel_size=LARGE_KERNEL_SIZE,
        small_kernel_size=SMALL_KERNEL_SIZE,
        n_res_blocks=N_RES_BLOCKS,
        scaling_factor=scaling_factor,
    )

    optimizer = torch.optim.Adam(model.parameters())

    load_checkpoint(
        model_filepath=MODEL_CHECKPOINT_PATH,
        state_filepath=STATE_CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    model.to(device)
    model.eval()

    if use_tiling:
        tile_size = 512
        overlap = 64

        if use_tta:
            logger.info(
                f"Starting tiled inference with TTA, tile={tile_size}, overlap={overlap}..."
            )
            num_transforms = len(tta_transforms)

            final_sr_canvas = torch.zeros(
                img_tensor.shape[0],
                img_tensor.shape[1],
                img_tensor.shape[2] * scaling_factor,
                img_tensor.shape[3] * scaling_factor,
                dtype=torch.float32,
            ).to("cpu")

            for tta_transform in tta_transforms:
                transformed_lr = tta_transform(img_tensor)

                sr_tta_result = upscale_tiled(
                    model=model,
                    lr_tensor=transformed_lr,
                    scaling_factor=scaling_factor,
                    tile_size=tile_size,
                    overlap=overlap,
                    device=device,
                )

                sr_tta_result_inversed = inverse_tta_transform(
                    sr_tta_result.to(device), tta_transform
                )

                final_sr_canvas += sr_tta_result_inversed.to("cpu")

            sr_image_tensor = final_sr_canvas / num_transforms
        else:
            logger.info(
                f"Starting tiled inference, tile={tile_size}, overlap={overlap}..."
            )
            sr_image_tensor = upscale_tiled(
                model=model,
                lr_tensor=img_tensor,
                scaling_factor=scaling_factor,
                tile_size=tile_size,
                overlap=overlap,
                device=device,
            )
    else:
        if use_tta:
            logger.info("Starting non-tiled inference with TTA...")
            num_transforms = len(tta_transforms)

            final_sr_canvas = torch.zeros(
                img_tensor.shape[0],
                img_tensor.shape[1],
                img_tensor.shape[2] * scaling_factor,
                img_tensor.shape[3] * scaling_factor,
                dtype=torch.float32,
            ).to("cpu")

            for tta_transform in tta_transforms:
                transformed_lr = tta_transform(img_tensor)

                with torch.inference_mode():
                    sr_tta_result = model(transformed_lr.to(device))

                sr_tta_result_inversed = inverse_tta_transform(
                    sr_tta_result, tta_transform
                )

                final_sr_canvas += sr_tta_result_inversed.to("cpu")

            sr_image_tensor = final_sr_canvas / num_transforms

        else:
            logger.info("Starting non-tiled inference...")
            with torch.inference_mode():
                sr_image_tensor = model(img_tensor.to(device))
            sr_image_tensor = sr_image_tensor.to("cpu")

    sr_image_tensor_device = sr_image_tensor.to(device)

    if comparison_image_path:
        logger.info("Creating comparison image...")
        compare_images(
            img_tensor.to(device),
            sr_image_tensor_device,
            comparison_image_path,
            scaling_factor,
        )

    sr_image_tensor_device = (sr_image_tensor_device + 1) / 2
    sr_image_tensor_device = sr_image_tensor_device.clamp(0, 1) * 255
    sr_image_tensor_device = sr_image_tensor_device.squeeze(0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sr_image = transforms.ToPILImage()(sr_image_tensor_device.byte().cpu())
    sr_image.save(output_path, format="PNG")

    logger.info(f"Upscaled image was saved to {output_path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    upscale_image(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        comparison_image_path=COMPARISON_IMAGE_PATH,
        scaling_factor=SCALING_FACTOR,
        use_tta=False,
        use_tiling=False,
        use_downscale=True,
        device=device,
    )


if __name__ == "__main__":
    main()
