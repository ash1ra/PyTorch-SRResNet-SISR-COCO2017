from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file
from torch import Tensor, nn, optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms

from config import create_logger


@dataclass
class Metrics:
    epochs: int = field(default=0)
    learning_rates: list[float] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    psnrs: list[float] = field(default_factory=list)
    ssims: list[float] = field(default_factory=list)


tta_transforms = [
    transforms.Compose([transforms.Lambda(lambda x: x)]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
    transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
        ]
    ),
]

logger = create_logger(log_level="INFO")


def transform_image(
    img_path: Path,
    scaling_factor: Literal[2, 4, 8],
    crop_size: int | None,
    test_mode: bool = False,
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(img_path.__fspath__())
    hr_img_tensor = img_tensor

    augmentation_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=(0, 90), expand=True),
        ]
    )

    # if not test_mode:
    #     hr_img_tensor = augmentation_transform(hr_img_tensor)

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

    normalize_transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    lr_img_tensor = lr_transform(hr_img_tensor)

    hr_img_tensor = normalize_transform(hr_img_tensor)
    lr_img_tensor = normalize_transform(lr_img_tensor)

    return hr_img_tensor, lr_img_tensor


def inverse_tta_transform(img_tensor: Tensor, transform: transforms.Compose) -> Tensor:
    for t in transform.transforms:
        if isinstance(t, transforms.RandomHorizontalFlip) or isinstance(
            t, transforms.RandomVerticalFlip
        ):
            img_tensor = t(img_tensor)

    return img_tensor


def save_checkpoint(
    model_filepath: str,
    state_filepath: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    metrics: Metrics,
    scaler: GradScaler | None = None,
    scheduler: MultiStepLR | None = None,
) -> None:
    Path(model_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(state_filepath).parent.mkdir(parents=True, exist_ok=True)

    save_file(model.state_dict(), model_filepath)

    state_dict = {
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": asdict(metrics),
    }

    if scaler:
        state_dict["scaler_state_dict"] = scaler.state_dict()

    if scheduler:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state_dict, state_filepath)

    logger.debug(f"The model's weights were saved after the {epoch} epoch")


def load_checkpoint(
    model_filepath: str,
    state_filepath: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    metrics: Metrics | None = None,
    scaler: GradScaler | None = None,
    scheduler: MultiStepLR | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
) -> int:
    if Path(model_filepath).exists() and Path(state_filepath).exists():
        model.load_state_dict(load_file(model_filepath, device=device))

        checkpoint_dict = torch.load(state_filepath, map_location=device)

        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

        if metrics and "metrics" in checkpoint_dict:
            metrics_dict = checkpoint_dict["metrics"]
            metrics.epochs = metrics_dict["epochs"]
            metrics.learning_rates = metrics_dict["learning_rates"]
            metrics.train_losses = metrics_dict["train_losses"]
            metrics.val_losses = metrics_dict["val_losses"]
            metrics.psnrs = metrics_dict["psnrs"]
            metrics.ssims = metrics_dict["ssims"]

        if scaler and "scaler_state_dict" in checkpoint_dict:
            scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint_dict:
            scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])

        return checkpoint_dict["epoch"]
    return 1


def compare_images(
    lr_img_tensor: Tensor,
    sr_img_tensor: Tensor,
    output_path: str | Path,
    scaling_factor: Literal[2, 4, 8] = 4,
) -> None:
    bicubic_label = "Bicubic"
    sr_label = "SRResNet"

    lr_img_tensor = (lr_img_tensor + 1) / 2
    lr_img_tensor = lr_img_tensor.clamp(0, 1) * 255
    if lr_img_tensor.dim() == 4:
        lr_img_tensor.squeeze_(0)

    sr_img_tensor = (sr_img_tensor + 1) / 2
    sr_img_tensor = sr_img_tensor.clamp(0, 1) * 255
    if sr_img_tensor.dim() == 4:
        sr_img_tensor.squeeze_(0)

    bicubic_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                size=(sr_img_tensor.shape[1], sr_img_tensor.shape[2]),
                interpolation=InterpolationMode.BICUBIC,
            ),
        ]
    )
    bicubic_img = bicubic_transform(lr_img_tensor.byte())

    sr_img = transforms.ToPILImage()(sr_img_tensor.byte())

    width, height = sr_img.size

    total_width = width
    total_height = height * 2 + 100
    comparison_img = Image.new("RGB", (total_width, total_height), color="white")

    comparison_img.paste(bicubic_img, (0, 50))
    comparison_img.paste(sr_img, (0, height + 100))

    draw = ImageDraw.Draw((comparison_img))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", size=36
        )
    except OSError:
        font = ImageFont.load_default()

    bicubic_text_width = draw.textlength(bicubic_label, font=font)
    sr_text_width = draw.textlength(sr_label, font=font)

    draw.text(
        ((width - bicubic_text_width) / 2, 5),
        bicubic_label,
        fill="black",
        font=font,
    )

    draw.text(
        ((width - sr_text_width) / 2, height + 55),
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


def plot_training_metrics(metrics: Metrics) -> None:
    epochs = list(range(0, metrics.epochs))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SRResNet Metrics", fontsize=16)

    axs[0, 0].plot(epochs, metrics.learning_rates, color="b")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Learning Rate")
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, metrics.train_losses, label="Train Loss", color="g")
    axs[0, 1].plot(epochs, metrics.val_losses, label="Val Loss", color="r")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(epochs, metrics.psnrs, color="r")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("PSNR")
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs, metrics.ssims, color="r")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("SSIM")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
