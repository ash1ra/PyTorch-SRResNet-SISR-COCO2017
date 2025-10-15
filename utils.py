from pathlib import Path

from torch import Tensor
from torchvision import transforms
from torchvision.io import decode_image


def transform_image(
    img_path: Path, scaling_factor: int, crop_size: int
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(img_path.__fspath__()) / 255.0 * 2.0 - 1.0

    hr_transform = transforms.RandomCrop(size=(crop_size, crop_size))
    lr_transform = transforms.Compose(
        [
            transforms.CenterCrop(size=(crop_size, crop_size)),
            transforms.Resize(
                size=(crop_size // scaling_factor, crop_size // scaling_factor),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    )

    hr_img_tensor = hr_transform(img_tensor)
    lr_img_tensor = lr_transform(img_tensor)

    return hr_img_tensor, lr_img_tensor
