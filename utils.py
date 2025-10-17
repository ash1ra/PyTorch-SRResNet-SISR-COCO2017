from pathlib import Path
from typing import Literal

import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file
from torch import Tensor, nn, optim
from torch.amp import GradScaler
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms


def transform_image(
    img_path: Path,
    scaling_factor: Literal[2, 4, 8],
    crop_size: int | None,
    test_mode: bool = False,
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(img_path.__fspath__())

    if test_mode:
        _, height, width = img_tensor.shape

        height_remainder = height % scaling_factor
        width_remainder = width % scaling_factor

        top = height_remainder // 2
        left = width_remainder // 2

        bottom = top + (height - height_remainder)
        right = left + (width - width_remainder)

        hr_img_tensor = img_tensor[:, top:bottom, left:right]
    elif crop_size:
        crop_transform = transforms.RandomCrop(size=(crop_size, crop_size))
        hr_img_tensor = crop_transform(img_tensor)

    normalize_transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    lr_transform = transforms.Compose(
        [
            transforms.Resize(
                size=(
                    hr_img_tensor.shape[1] // scaling_factor,
                    hr_img_tensor.shape[2] // scaling_factor,
                ),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    )

    lr_img_tensor = lr_transform(hr_img_tensor)

    hr_img_tensor = normalize_transform(hr_img_tensor)
    lr_img_tensor = normalize_transform(lr_img_tensor)

    return hr_img_tensor, lr_img_tensor


def save_checkpoint(
    model_filepath: str,
    state_filepath: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None = None,
) -> None:
    Path(model_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(state_filepath).parent.mkdir(parents=True, exist_ok=True)

    save_file(model.state_dict(), model_filepath)

    state_dict = {
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scaler:
        state_dict["scaler_state_dict"] = scaler.state_dict()

    torch.save(state_dict, state_filepath)


def load_checkpoint(
    model_filepath: str,
    state_filepath: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
) -> int:
    if Path(model_filepath).exists() and Path(state_filepath).exists():
        model.load_state_dict(load_file(model_filepath, device=device))

        checkpoint_dict = torch.load(state_filepath, map_location=device)

        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint_dict:
            scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])

        return checkpoint_dict["epoch"]
    return 1


def compare_images(
    hr_img_tensor: Tensor,
    sr_img_tensor: Tensor,
    output_path: str | Path,
) -> None:
    hr_label = "Original"
    sr_label = "Upscaled"

    hr_img_tensor = (hr_img_tensor + 1) / 2
    hr_img_tensor = hr_img_tensor.clamp(0, 1) * 255
    hr_img_tensor.squeeze_(0)

    sr_img_tensor = (sr_img_tensor + 1) / 2
    sr_img_tensor = sr_img_tensor.clamp(0, 1) * 255
    sr_img_tensor.squeeze_(0)

    to_pil_transform = transforms.ToPILImage()
    hr_img = to_pil_transform(hr_img_tensor.byte())
    sr_img = to_pil_transform(sr_img_tensor.byte())

    width, height = hr_img.size
    sr_img = sr_img.resize((width, height), Image.Resampling.BICUBIC)

    total_width = width
    total_height = height * 2 + 100
    comparison_img = Image.new("RGB", (total_width, total_height), color="white")

    comparison_img.paste(hr_img, (0, 50))
    comparison_img.paste(sr_img, (0, height + 100))

    draw = ImageDraw.Draw(comparison_img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", size=18
        )
    except OSError:
        font = ImageFont.load_default()

    draw.text(
        (width // 2 - len(hr_label) * 5, 15),
        hr_label,
        fill="black",
        font=font,
    )
    draw.text(
        (width // 2 - len(sr_label) * 5, height + 65),
        sr_label,
        fill="black",
        font=font,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_img.save(output_path, format="PNG")


def rgb_to_ycbcr(image_tensor: torch.Tensor) -> torch.Tensor:
    if image_tensor.dim() == 4:
        image_tensor.squeeze_(0)

    image_tensor = (image_tensor + 1) / 2

    weights = torch.tensor(
        [0.299, 0.587, 0.114],
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    )

    Y_channel = torch.sum(
        image_tensor * weights.view(1, 3, 1, 1),
        dim=1,
        keepdim=True,
    )

    return Y_channel


def format_time(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
