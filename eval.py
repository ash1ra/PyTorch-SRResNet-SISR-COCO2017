from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms.v2 as transforms
from torchvision.io import decode_image

from model import SRResNet
from utils import load_checkpoint

INPUT_PATH = Path.home() / "Downloads/1.jpg"
OUTPUT_PATH = Path("result.png")
SCALING_FACTOR: Literal[2, 4, 8] = 4
N_CHANNELS = 64
N_RES_BLOCKS = 16
LARGE_KERNEL_SIZE = 9
SMALL_KERNEL_SIZE = 3

CHECKPOINTS_DIR = Path("checkpoints")
MODEL_NAME = "srresnet"
MODEL_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_model.safetensors"
STATE_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_state.pth"


def upscale_image(
    input_path: Path,
    output_path: Path,
    scaling_factor: Literal[2, 4, 8],
    device: Literal["cpu", "cuda"] = "cpu",
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if input_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("Input image must be in JPG or PNG format")

    model = SRResNet(
        n_channels=N_CHANNELS,
        large_kernel_size=LARGE_KERNEL_SIZE,
        small_kernel_size=SMALL_KERNEL_SIZE,
        n_res_blocks=N_RES_BLOCKS,
        scaling_factor=scaling_factor,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    load_checkpoint(
        MODEL_CHECKPOINT_PATH,
        STATE_CHECKPOINT_PATH,
        model,
        optimizer,
        device=device,
    )

    model.eval()

    input_path = Path(input_path)
    img_tensor = decode_image(input_path.__fspath__())

    if img_tensor.shape[0] > 3:
        img_tensor = img_tensor[:3]

    transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img_tensor = transform(img_tensor).unsqueeze(0).to(device)

    with torch.inference_mode():
        upscaled_tensor = model(img_tensor)

    upscaled_tensor = (upscaled_tensor + 1) / 2
    upscaled_tensor = upscaled_tensor.clamp(0, 1) * 255
    upscaled_tensor = upscaled_tensor.squeeze(0).byte()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pil_image = transforms.ToPILImage()(upscaled_tensor)
    pil_image.save(output_path, format="PNG")

    print(f"Upscaled image saved to {output_path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    upscale_image(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        scaling_factor=SCALING_FACTOR,
        device=device,
    )


if __name__ == "__main__":
    main()
