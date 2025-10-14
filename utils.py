from pathlib import Path

from torch import Tensor
from torchvision import transforms
from torchvision.io import decode_image


def transform_image(img_path: Path, scaling_factor: int) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(img_path.__fspath__())

    crop_height = (img_tensor.shape[1] // scaling_factor) * scaling_factor
    crop_width = (img_tensor.shape[2] // scaling_factor) * scaling_factor

    hr_transform = transforms.RandomCrop((crop_height, crop_width))
    lr_transform = transforms.Compose(
        [
            transforms.CenterCrop((crop_height, crop_width)),
            transforms.Resize(
                (crop_height // scaling_factor, crop_width // scaling_factor),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ]
    )

    hr_img_tensor = hr_transform(img_tensor)
    lr_img_tensor = lr_transform(img_tensor)

    return hr_img_tensor, lr_img_tensor
