# PyTorch SRResNet for SISR task

This project implements a **SRResNet** (Super-Resolution Residual Network) model for **SISR** (Single Image Super-Resolution) task. The primary goal is to upscale low-resolution (LR) images by a given factor (2x, 4x, 8x) to produce super-resolution (SR) images with high fidelity and perceptual quality.

This implementation is based on the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) and inspired by the [sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution) tutorial.

## Demonstration

The following images compare the standard bicubic interpolation with the output of the SRResNet model.

![Comparison image](images/comparison_img.png)

## Key Features

- **Efficient Upsampling:** Uses **Sub-pixel Convolution (PixelShuffle)** instead of Transposed Convolution for cleaner upscaling and fewer artifacts.
- **Fast Convergence:** Employs the **OneCycleLR** scheduler to accelerate training.
- **Optimized Training:** Leverages **AMP (Automatic Mixed Precision)** for faster computation on compatible GPUs.
- **Data Augmentation:** Applies random flips and rotations to training images to improve model generalization.
- **Accurate Evaluation:**
    * Uses **TTA (Test-Time Augmentation)** for more robust and accurate results during testing and inference.
    * Calculates metrics (PSNR, SSIM) correctly on the **Y-channel** (luminance) after converting from RGB.
    * Implements **side pixel cropping** (shaving) before metric calculation, which is a standard practice to avoid boundary effects.
- **Robust Checkpointing:**
    * Saves the model with the best PSNR score separately.
    * Saves a checkpoint after every epoch.
    * Safely saves the current state upon **Keyboard Interrupt (Ctrl+C)**.
- **Comprehensive Logging:**
    * Logs all training progress to both the console and a timestamped file in the `logs/` directory.
    * Automatically generates and saves training plots (Loss, PSNR, SSIM, Learning Rate) upon completion.

## Model Architecture

```ascii
                                     Input (LR Image)
                                            |
                                            v
                        +-Input-Conv-Block-----------------------+
                        | Conv2D (9x9 kernel) (3 -> 64 channels) |
                        | PReLU                                  |
                        +----------------------------------------+
                                            |
                                            +---------------------------+
                                            |                           |
                                            v                           |
                  +-----+-16x-Residual-Blocks---------------------+     |
                  |     | Conv2D (3x3 kernel) (64 -> 64 channels) |     |
                  |     | Batch Normalization                     |     |
(Skip connection) |     | PReLU                                   |     | (Skip connection)
                  |     | Conv2D (3x3 kernel) (64 -> 64 channels) |     |
                  |     | Batch Normalization                     |     |
                  +-----+-----------------------------------------+     |
                                            |                           |
                                            v                           |
                        +-Middle-Conv-Block-----------------------+     |
                        | Conv2D (3x3 kernel) (64 -> 64 channels) |     |
                        | Batch Normalization                     |     |
                        +-----------------------------------------+     |    
                                            |                           |
                                            +---------------------------+
                                            |
                                            v
                        +-2x-Sub-pixel-Conv-Blocks-----------------+
                        | Conv2D (3x3 kernel) (64 -> 256 channels) |
                        | PixelShuffle (h, w, 256 -> 2h, 2w, 64)   |
                        | PReLU                                    |
                        +------------------------------------------+
                                            |
                                            v
                        +-Final-Conv-Block-----------------------+
                        | Conv2D (9x9 kernel) (64 -> 3 channels) |
                        | Tanh                                   |
                        +----------------------------------------+
                                            |
                                            v
                                     Output (SR Image)
```

## Datasets

- **Training:** The model is trained on the **COCO 2017** (`2017 Train images [118K/18GB]`) dataset. The `data_processing.py` script dynamically creates LR images from HR images using bicubic downsampling and applies random crops and augmentations (flips, rotations).
- **Validation:** A test dataset (`2017 Test images [41K/6GB]`) is used for validation instead of validation set (`2017 Val images [5K/1GB]`) because it contains more images.
- **Testing:** The `test.py` script is configured to evaluate the trained model on standard benchmark datasets: **Set5**, **Set14**, **BSDS100**, and **Urban100**.

## Project Structure

```
.
├── checkpoints/             # Stores model weights (.safetensors) and training states (.pth)
├── images/                  # Directory for inference inputs, outputs, and training plots
├── config.py                # Configures the application logger
├── data_processing.py       # Defines the SRDataset class and image transformations
├── inference.py             # Script to run the model on a single image
├── model.py                 # SRResNet model architecture definition
├── test.py                  # Script for evaluating the model on benchmark datasets
├── train.py                 # Script for training the model
└── utils.py                 # Utility functions (metrics, TTA, checkpoints, plotting)
```

## Configuration

Most hyperparameters and settings are defined as constants at the top of the `train.py`, `test.py`, and `inference.py` files.

Key settings in `train.py`:
- `SCALING_FACTOR`: 4
- `CROP_SIZE`: 128
- `N_RES_BLOCKS`: 16
- `BATCH_SIZE`: 32
- `LEARNING_RATE`: 1e-5 (initial LR for OneCycleLR)
- `MAX_LEARNING_RATE`: 1e-4
- `EPOCHS`: 500
- `NUM_WORKERS`: 8
- `LOAD_MODEL`: Set to `True` to resume training from a checkpoint.
- `TRAIN_DATASET_PATH`: Path to the train folder.
- `VAL_DATASET_PATH`: Path to the validation folder.
- `DEV_MODE`: Set to `True` to use 1280 images instead of whole dataset.

## Setting Up and Running the Project

### 1. Installation

1. Clone the repository:
```bash
git clone https://github.com/ash1ra/SRResNet-SISR-COCO2017.git
cd SRResNet-SISR-COCO2017
```

2. Create `.venv` and install dependencies:
```bash
uv sync
```

3. Activate a virtual environment:
```bash
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

### 2. Data Preparation

1.  [Download](https://cocodataset.org/#download) the **COCO 2017** datasets.
2.  [Download](https://figshare.com/articles/dataset/BSD100_Set5_Set14_Urban100/21586188) the standard benchmark datasets (**Set5**, **Set14**, **BSDS100**, **Urban100**).
3.  Organize your data directory as expected by the scripts:
    ```
    data/
    ├── COCO2017_train/
    │   ├── 000000000009.jpg
    │   └── ...
    ├── COCO2017_test/
    │   ├── 000000000139.jpg
    │   └── ...
    ├── Set5/
    │   ├── baboon.png
    │   └── ...
    ├── Set14/
    │   └── ...
    ```
4.  Update the paths (`TRAIN_DATASET_PATH`, `VAL_DATASET_PATH`, `DATASETS_DIR`) in `train.py` and `test.py` to match your data structure.

### 3. Training

1.  Adjust parameters in `train.py` as needed.
2.  Run the training script:
    ```bash
    python train.py
    ```
3.  Training progress will be logged to the console and to a file in the `logs/` directory.
4.  Checkpoints will be saved in `checkpoints/`. A plot of the training metrics will be saved in `images/` upon completion.

### 4. Testing

To evaluate the model's performance on the test datasets:

1.  Ensure the `MODEL_CHECKPOINT_PATH` in `test.py` points to your trained model (e.g., `srresnet_model_best.safetensors`).
2.  Run the test script:
    ```bash
    python test.py
    ```
3.  The script will print the average PSNR and SSIM for each dataset.

### 5. Inference

To upscale a single image:

1.  Place your image in the `images/` folder (or update the path).
2.  In `inference.py`, set `INPUT_PATH` to your image and `MODEL_CHECKPOINT_PATH` to your trained model.
3.  Run the script:
    ```bash
    python inference.py
    ```
4.  The upscaled image (`sr_img_*.png`) and a comparison image (`comparison_img_*.png`) will be saved in the `images/` directory.

## Training Results

![The following chart shows the progression of loss, learning rate, PSNR, and SSIM during training.](images/training_metrics_final.png)
![The following chart shows the progression of loss, learning rate, PSNR, and SSIM during training.](images/training_metrics_final_zoomed.png)

The model was trained for 500 epochs with a batch size of 32 on an NVIDIA RTX 4060 Ti (8 GB) and took 59 hours. The rest of the hyperparameters are specified on the chart. The final model is the one with the highest PSNR value.

## Benchmark Evaluation (4x Upscaling)

The final model (`srresnet_model_best.safetensors`) was evaluated on standard benchmark datasets using TTA. Metrics are calculated on the Y-channel after shaving 4px (the scaling factor) from the border.

The results are compared against the original paper's SRResNet and the sgrvinod tutorial implementation.

**PSNR (dB) Comparison**
| Dataset / Implementation | SRResNet (this project) | SRResNet (sgrvinod) | SRResNet (paper)
| :--- | :---: | :---: | :---: |
| **Set5** | 31.0983 | 31.927 | 32.05 |
| **Set14** | 27.5783 | 28.588 | 28.49 |
| **BSDS100** | 26.4378 | 27.587 | 27.58 |
| **Urban100**| 25.3613 | — | — |

**SSIM Comparison**
| Dataset / Implementation | SRResNet (this project) | SRResNet (sgrvinod) | SRResNet (paper)
| :--- | :---: | :---: | :---: |
| **Set5** | 0.8849 | 0.902  | 0.9019 |
| **Set14** | 0.7705 | 0.799 | 0.8184 |
| **BSDS100** | 0.7232 | 0.756 | 0.7620 |
| **Urban100** | 0.7889 | — | — |

***Note:*** *My results might be slightly different from the paper's, which is expected. The paper's authors may have used different training datasets (ImageNet vs COCO), different training durations, or minor variations in implementation.*

***Note 2:*** *It's important to remember that in Super-Resolution, traditional metrics like PSNR and SSIM are not the only measure of success. As highlighted in the tutorial and the original paper, a model (like this SRResNet) trained to minimize MSE will maximize PSNR, but this often results in overly smooth images that lack fine, realistic textures. Perceptually-driven models (like SRGAN) often score lower on PSNR/SSIM but produce results that look much more convincing to the human eye.*

## Visual Comparisons

The following images compare the standard bicubic interpolation with the output of the SRResNet model. I tried to use different images that would be more clearly seen that the model handles well with anime/cartoon images and quite bad with photorealistic.

![Comparisson image 1](images/comparison_img_1.png)
![Comparisson image 2](images/comparison_img_2.png)
![Comparisson image 3](images/comparison_img_3.png)
![Comparisson image 4](images/comparison_img_4.png)
![Comparisson image 5](images/comparison_img_5.png)

## Acknowledgements
- This project is heavily inspired by the excellent [a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution) by sgrvinod.
- Based on the paper: [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

