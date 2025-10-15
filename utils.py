from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms


def transform_image(
    img_path: Path, scaling_factor: int, crop_size: int
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(img_path.__fspath__())

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
