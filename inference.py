from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms.v2 as transforms
from torchvision.io import decode_image

from config import create_logger
from model import SRResNet
from utils import compare_images, inverse_tta_transform, load_checkpoint, tta_transforms

# INPUT_PATH = Path("images/5.jpg")
# OUTPUT_PATH = Path("images/result_5.png")
# COMPARISON_IMAGE_PATH = Path("images/comparison_image_5.png")

INPUT_PATH = Path("data/Set14/baboon.png")
OUTPUT_PATH = Path("images/sr_baboon.png")
COMPARISON_IMAGE_PATH = Path("images/comparison_baboon.png")

SCALING_FACTOR: Literal[2, 4, 8] = 4
N_CHANNELS = 96
N_RES_BLOCKS = 16
LARGE_KERNEL_SIZE = 9
SMALL_KERNEL_SIZE = 3

CHECKPOINTS_DIR = Path("checkpoints")
MODEL_NAME = "srresnet"
MODEL_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_model.safetensors"
STATE_CHECKPOINT_PATH = CHECKPOINTS_DIR / f"{MODEL_NAME}_state.pth"

logger = create_logger(log_level="INFO")


def upscale_image(
    input_path: Path,
    output_path: Path,
    comparison_image_path: Path | None,
    scaling_factor: Literal[2, 4, 8] = 4,
    use_tta: bool = True,
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
        model_filepath=MODEL_CHECKPOINT_PATH,
        state_filepath=STATE_CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
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
        if use_tta:
            sr_images = []
            for tta_transform in tta_transforms:
                sr_image_tensor = model(tta_transform(img_tensor))
                sr_image_tensor = inverse_tta_transform(sr_image_tensor, tta_transform)
                sr_images.append(sr_image_tensor)
            sr_image_tensor = torch.mean(torch.stack(sr_images), dim=0)
        else:
            sr_image_tensor = model(img_tensor)

    if comparison_image_path:
        compare_images(
            img_tensor, sr_image_tensor, comparison_image_path, scaling_factor
        )

    sr_image_tensor = (sr_image_tensor + 1) / 2
    sr_image_tensor = sr_image_tensor.clamp(0, 1) * 255
    sr_image_tensor = sr_image_tensor.squeeze(0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sr_image = transforms.ToPILImage()(sr_image_tensor.byte())
    sr_image.save(output_path, format="PNG")

    logger.info(f"Upscaled image was saved to {output_path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    upscale_image(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        comparison_image_path=COMPARISON_IMAGE_PATH,
        scaling_factor=SCALING_FACTOR,
        use_tta=True,
        device=device,
    )


if __name__ == "__main__":
    main()
