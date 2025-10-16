from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn, optim
from torch.amp import GradScaler
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms


def transform_image(
    img_path: Path,
    scaling_factor: Literal[2, 4, 8],
    crop_size: int,
    test_mode: bool = False,
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(img_path.__fspath__())

    if test_mode:
        hr_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(crop_size, crop_size)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        hr_transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(crop_size, crop_size)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    lr_transform = transforms.Compose(
        [
            transforms.CenterCrop(size=(crop_size, crop_size)),
            transforms.Resize(
                size=(crop_size // scaling_factor, crop_size // scaling_factor),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    hr_img_tensor = hr_transform(img_tensor)
    lr_img_tensor = lr_transform(img_tensor)

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

    state_dict = {
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scaler:
        state_dict["scaler_state_dict"] = scaler.state_dict()

    save_file(model.state_dict(), model_filepath)
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
