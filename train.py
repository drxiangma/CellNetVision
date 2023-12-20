import argparse
from modeling import CustomCNN, AlexNet, ResNet, VisionTransformer
from curriculum_learning import apply_curriculum_learning
from data_processing import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time

def train_model(model_type, num_conv_layers, curriculum, image_folder, ground_truth_folder,
                lr, weight_decay, batch_size, epochs, patience):
    if model_type == 'CustomCNN':
        model = CustomCNN(num_conv_layers=num_conv_layers)  # Create Custom CNN model with selected conv layers
    elif model_type == 'AlexNet':
        model = AlexNet()  # Load pre-trained AlexNet
    elif model_type == 'ResNet':
        model = ResNet()  # Load pre-trained ResNet
    elif model_type == 'VisionTransformer':
        model = VisionTransformer()  # Load pre-trained Vision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare data from image
    train_data_loader, val_data_loader, _ = load_dataset(model, image_folder, 0.8, batch_size, curriculum)

    # Train the selected model using prepared data
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_model_path = f'models/best_{model_type}_model.pth'  # Path to save the best model
    counter = 0  # Counter for early stopping
    num_epoch = 0
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_epoch += 1
        for images, labels, _ in train_data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)

        mean_train_loss = total_train_loss / len(train_data_loader.dataset)
        train_losses.append(mean_train_loss)

        # Validation loop (evaluate the model)
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for images, labels, _ in val_data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # total_val_loss += torch.abs(outputs - labels.view(-1, 1).float()).sum().item()
                total_val_loss += criterion(outputs, labels.view(-1, 1).float()).item() * images.size(0)

            mean_val_loss = total_val_loss / len(train_data_loader.dataset)
            val_losses.append(mean_val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {mean_train_loss:.4f}, Val Loss: {mean_val_loss:.4f}")

            # Check for early stopping
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                counter = 0
                # Save the model with the best validation loss
                model.save(best_model_path)
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1} as validation loss did not improve.")
                break

    end_time = time.time()
    print(f'time per epoch: {(end_time - start_time) / num_epoch}')

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train models for cell counting')
    parser.add_argument('--model', type=str, choices=['CustomCNN', 'AlexNet', 'ResNet', 'VisionTransformer'],
                        help='Select the model type for training')
    parser.add_argument('--num-conv-layers', type=int, default=1, choices=range(1, 5),
                        help='Select the number of convolutional layers for CustomCNN (1-4)')
    parser.add_argument('--curriculum', action='store_true', help='Apply curriculum learning')
    parser.add_argument('--image-folder', type=str, help='Path to the folder containing images for training and validation')
    parser.add_argument('--ground-truth-folder', type=str, help='Path to the folder containing ground truth data')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for regularization')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')

    args = parser.parse_args()

    train_model(args.model, args.num_conv_layers, args.curriculum, args.image_folder, args.ground_truth_folder,
                args.learning_rate, args.weight_decay, args.batch_size, args.epochs, args.patience)

if __name__ == "__main__":
    main()
