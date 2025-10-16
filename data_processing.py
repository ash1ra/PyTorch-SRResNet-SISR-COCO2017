from pathlib import Path
from typing import Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from utils import transform_image


class SRDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        scaling_factor: Literal[2, 4, 8],
        crop_size: int,
        test_mode: bool = False,
        dev_mode: bool = False,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.test_mode = test_mode
        self.images = []

        for img_path in Path(data_folder).iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                with Image.open(img_path) as img:
                    width, height = img.size
                    if (
                        img.mode == "RGB"
                        and width >= self.crop_size
                        and height >= self.crop_size
                    ):
                        self.images.append(img_path)

        if dev_mode:
            self.images = self.images[:1280]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return transform_image(
            self.images[i],
            scaling_factor=self.scaling_factor,
            crop_size=self.crop_size,
            test_mode=self.test_mode,
        )
