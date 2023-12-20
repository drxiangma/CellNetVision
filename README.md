# CellNetVision

Welcome to CellNetVision, a repository containing various models and tools for cell counting using deep learning techniques. This repository offers a comprehensive collection of functions utilizing a range of models, including customized CNNs, pretrained AlexNet, ResNet, and Vision Transformer, employing techniques like curriculum learning, hyperparameter fine-tuning, and cross-validation, for accurate cell counting. If you feel that this repository is helpful, please give it a star! Thanks!

## Dataset Overview
The dataset consists of images in TIFF format and corresponding CSV files each row of which contains coordinates for each cell in the images. The images are stored in the 'images' folder, while the CSV files reside in the 'ground_truth' folder.

### Dataset Structure
- **Data/images/**: Contains TIFF images of cells.
- **Data/ground_truth/**: Contains CSV files corresponding to the images, each file contains cell coordinates.

## Files Overview

### train.py
- Trains models for cell counting using various architectures.
- **Flags**:
  - `--model`: Choose from ['CustomCNN', 'AlexNet', 'ResNet', 'VisionTransformer'] for model selection.
  - `--num-conv-layers`: Select the number of convolutional layers for CustomCNN (1-4).
  - `--curriculum`: Apply curriculum learning.
  - `--image-folder`: Path to the folder containing total images.
  - `--ground-truth-folder`: Path to the folder containing ground truth data.
  - `--learning-rate`: Learning rate for training (default: 0.001).
  - `--weight-decay`: Weight decay for regularization (default: 0.0).
  - `--batch-size`: Batch size for training (default: 32).
  - `--epochs`: Number of epochs for training (default: 10).
  - `--patience`: Patience for early stopping (default: 3).

### test.py
- Evaluates trained models using test data.
- **Flags**:
  - `--model-path`: Path to the trained model.
  - `--image-folder`: Path to the folder containing total images.

### modeling.py
- Contains model classes for CustomCNN, AlexNet, ResNet, and Vision Transformer.

### data_processing.py
- Includes the `Data_Generator` class and `load_dataset` function for dataset preparation.

### finetune_hyperparameters.py
- Performs hyperparameter tuning for models.
- **Flags**:
  - `--model`: Choose from ['CustomCNN', 'AlexNet', 'ResNet', 'VisionTransformer'] for model selection.
  - `--num-conv-layers`: Select the number of convolutional layers for CustomCNN (1-4).
  - `--image-folder`: Path to the folder containing images for testing.

### k-fold_cross_validation.py
- Executes k-fold cross-validation for model evaluation.
- **Flags**:
  - `--model`: Choose from ['CustomCNN', 'AlexNet', 'ResNet', 'VisionTransformer'] for model selection.
  - `--num-conv-layers`: Select the number of convolutional layers for CustomCNN (1-4).
  - `--image-folder`: Path to the image folder.
  - `--k-fold`: Number of folds for cross-validation.
  - `--batch-size`: Batch size.
  - `--epochs`: Number of epochs.
  - `--learning-rate`: Learning rate.
  - `--weight-decay`: Weight decay.

### augmentation_flipping_erasing.py
- Provides functions for flipping and random erasing of images.

## How to Use

1. **Training**:
    - Run `train.py` with appropriate flags to train a model on the dataset.
2. **Testing**:
    - Use `test.py` with the model path to evaluate the model on test images.
3. **Hyperparameter Tuning**:
    - Utilize `finetune_hyperparameters.py` to fine-tune model hyperparameters.
4. **Cross-validation**:
    - Perform k-fold cross-validation using `k-fold_cross_validation.py`.
5. **Data Augmentation**:
    - Use functions from `augmentation_flipping_erasing.py` for image augmentation.

## Usage Example
Training with CustomCNN:

python train.py --model CustomCNN --num-conv-layers 3 --curriculum --image-folder path/to/images --ground-truth-folder path/to/ground_truth --learning-rate 0.001 --batch-size 32 --epochs 15 --patience 5

Please replace 'path/to/images' and 'path/to/ground_truth' with the actual paths to your image and ground truth folders, respectively. Adjust the usage example as per your requirements and folder structure.

Feel free to explore and leverage different functionalities of this repository for cell counting tasks!
