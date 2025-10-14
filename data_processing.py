from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from utils import transform_image


class SRDataset(Dataset):
    def __init__(self, data_folder: str, scaling_factor: int) -> None:
        self.scaling_factor = scaling_factor
        self.images = list(Path(data_folder).iterdir())

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return transform_image(self.images[i], scaling_factor=self.scaling_factor)


if __name__ == "__main__":
    dataset = SRDataset("./data/Set5/", scaling_factor=4)
    print(dataset[0][1])
