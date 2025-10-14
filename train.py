from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_processing import SRDataset
from model import SRResNet

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 3


def train_step(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: Literal["cpu", "cuda"] = "cpu",
) -> float:
    train_loss = 0
    model.train()

    for i, (hr_image_tensor, lr_image_tensor) in enumerate(data_loader):
        hr_image_tensor = hr_image_tensor.to(device)
        lr_image_tensor = lr_image_tensor.to(device)

        preds = model(lr_image_tensor)

        loss = loss_fn(preds, hr_image_tensor)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)

    return train_loss


def train(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    device: Literal["cpu", "cuda"] = "cpu",
) -> None:
    for epoch in range(1, epochs + 1):
        train_loss = train_step(data_loader, model, loss_fn, optimizer, device)
        print(f"Epoch: {epoch} | Train loss: {train_loss}")


def main() -> None:
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    dataset = SRDataset(
        data_folder="data/COCO2017_train", scaling_factor=4, min_size=256
    )
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SRResNet(
        n_channels=64,
        large_kernel_size=9,
        small_kernel_size=3,
        n_res_blocks=16,
        scaling_factor=4,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(data_loader, model, loss_fn, optimizer, EPOCHS, device)


if __name__ == "__main__":
    main()
