from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing import SRDataset
from model import SRResNet
from utils import load_checkpoint, save_checkpoint

SCALING_FACTOR: Literal[2, 4, 8] = 4
CROP_SIZE = 96

N_CHANNELS = 64
N_RES_BLOCKS = 16
LARGE_KERNEL_SIZE = 9
SMALL_KERNEL_SIZE = 3

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
LOAD_MODEL = True

NUM_WORKERS = 8

CHECKPOINTS_DIR = Path("checkpoints")
MODEL_NAME = "srresnet"
MODEL_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_model.safetensors"
STATE_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_state.pth"


def train_step(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None,
    device: Literal["cpu", "cuda"] = "cpu",
) -> float:
    train_loss = 0
    model.train()

    for i, (hr_image_tensor, lr_image_tensor) in enumerate(
        tqdm(data_loader, desc="Training", leave=False)
    ):
        hr_image_tensor = hr_image_tensor.to(device, non_blocking=True)
        lr_image_tensor = lr_image_tensor.to(device, non_blocking=True)

        if scaler:
            with autocast(device):
                preds = model(lr_image_tensor)
                loss = loss_fn(preds, hr_image_tensor)
        else:
            preds = model(lr_image_tensor)
            loss = loss_fn(preds, hr_image_tensor)

        train_loss += loss.item()

        optimizer.zero_grad()

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    train_loss /= len(data_loader)

    return train_loss


def train(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    start_epoch: int,
    epochs: int,
    scaler: GradScaler | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
) -> None:
    try:
        progress_bar = tqdm(range(start_epoch, epochs + 1), desc="Epochs", leave=True)

        for epoch in progress_bar:
            train_loss = train_step(
                data_loader, model, loss_fn, optimizer, scaler, device
            )
            progress_bar.set_postfix({"loss": f"{train_loss:.4f}"})
            print()

            save_checkpoint(
                MODEL_CHECKPOINT_PATH,
                STATE_CHECKPOINT_PATH,
                epoch,
                model,
                optimizer,
                scaler,
            )
    except KeyboardInterrupt:
        save_checkpoint(
            MODEL_CHECKPOINT_PATH,
            STATE_CHECKPOINT_PATH,
            epoch,
            model,
            optimizer,
            scaler,
        )
        exit(0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SRDataset(
        data_folder="data/COCO2017_train",
        scaling_factor=SCALING_FACTOR,
        crop_size=CROP_SIZE,
        # dev_mode=True,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True if device == "cuda" else False,
        num_workers=NUM_WORKERS,
    )

    model = SRResNet(
        n_channels=N_CHANNELS,
        large_kernel_size=LARGE_KERNEL_SIZE,
        small_kernel_size=SMALL_KERNEL_SIZE,
        n_res_blocks=N_RES_BLOCKS,
        scaling_factor=SCALING_FACTOR,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device)

    if LOAD_MODEL:
        start_epoch = load_checkpoint(
            MODEL_CHECKPOINT_PATH,
            STATE_CHECKPOINT_PATH,
            model,
            optimizer,
            scaler,
            device,
        )
    else:
        start_epoch = 1

    train(
        data_loader=data_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        start_epoch=start_epoch,
        epochs=EPOCHS,
        scaler=scaler,
        device=device,
    )


if __name__ == "__main__":
    main()
