from pathlib import Path

from torch import Tensor
from torchvision import transforms
from torchvision.io import decode_image


def transform_image(
    img_path: Path, scaling_factor: int, min_size: int
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(img_path.__fspath__()).float() / 255.0

    hr_transform = transforms.RandomCrop((min_size, min_size))
    lr_transform = transforms.Compose(
        [
            transforms.CenterCrop((min_size, min_size)),
            transforms.Resize(
                (min_size // scaling_factor, min_size // scaling_factor),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ]
    )

    hr_img_tensor = hr_transform(img_tensor)
    lr_img_tensor = lr_transform(img_tensor)

    return hr_img_tensor, lr_img_tensor
