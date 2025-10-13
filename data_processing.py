from pathlib import Path

from torch.utils.data import Dataset


class SRDataset(Dataset):
    def __init__(self, data_folder: str) -> None:
        self.images = list(Path(data_folder).iterdir())

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> ...:
        # returns 2 images: a low resolution version (lr) and a target (hr)
        # lr can be created by using function from the utils file
        return self.images[i]


if __name__ == "__main__":
    dataset = SRDataset("./data/Set5/")
    print(dataset[0])
