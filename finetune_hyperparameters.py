import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_processing import load_dataset
from modeling import CustomCNN, AlexNet, ResNet, VisionTransformer
import argparse

def train_and_test_model(model_type, num_conv_layers, batch_size, lr, weight_decay, patience, train_data_loader, val_data_loader, test_data_loader):
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 100
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        total_train_loss = 0
        sample_size = 0
        for images, labels, _ in train_data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            sample_size += images.size(0)

        mean_train_loss = total_train_loss / sample_size
        train_losses.append(mean_train_loss)

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            sample_size = 0
            for images, labels, _ in val_data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_val_loss += criterion(outputs, labels.view(-1, 1).float()).item()
                sample_size += images.size(0)

            mean_val_loss = total_val_loss / sample_size
            val_losses.append(mean_val_loss)

            # Check for early stopping
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1} as validation loss did not improve.")
                break

    # Testing loop
    threshold = 5.0
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        test_loss_mae = 0
        test_loss_mse = 0

        for images, labels, _ in test_data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_samples += images.size(0)

            test_loss_mae += torch.abs(outputs - labels.view(-1, 1).float()).sum().item()
            test_loss_mse += criterion(outputs, labels.view(-1, 1).float()).item()

            percentage_difference = torch.abs((outputs - labels.view(-1, 1).float()) / labels.view(-1, 1).float()) * 100
            total_correct += torch.sum(percentage_difference <= threshold)

        mean_mae = test_loss_mae / total_samples
        mean_mse = test_loss_mse / total_samples
        accuracy = total_correct / total_samples

        print(f"Batch Size: {batch_size}, Learning Rate: {lr}, Weight Decay: {weight_decay}, Patience: {patience}")
        print(f"MAE: {mean_mae:.4f}, MSE: {mean_mse:.4f}, Accuracy: {accuracy:.4f}")

        return accuracy

def main():
    parser = argparse.ArgumentParser(description='Train models for cell counting')
    parser.add_argument('--model', type=str, choices=['CustomCNN', 'AlexNet', 'ResNet', 'VisionTransformer'],
                        help='Select the model type for training')
    parser.add_argument('--num-conv-layers', type=int, default=1, choices=range(1, 5),
                        help='Select the number of convolutional layers for CustomCNN (1-4)')
    parser.add_argument('--image-folder', type=str, help='Path to the folder containing images for testing')
    args = parser.parse_args()
    
    # Define hyperparameters to tune
    batch_sizes = [16, 32, 64, 128]
    learning_rates = [0.0001, 0.0005, 0.001]
    weight_decays = [0.0001, 0.0005, 0.001]
    patience_values = [3, 5, 10]

    best_accuracy = 0.0
    best_hyperparameters = {}

    # Iterate through combinations of hyperparameters
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for weight_decay in weight_decays:
                for patience in patience_values:
                    # Prepare data from image
                    train_data_loader, val_data_loader, test_data_loader = load_dataset(args.image_folder, 0.8, batch_size)

                    # Train and test model with current hyperparameters
                    accuracy = train_and_test_model(args.model, args.num_conv_layers, args.image_folder, batch_size, lr, weight_decay, patience,
                                                    train_data_loader, val_data_loader, test_data_loader)

                    # Check for best hyperparameters and accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_hyperparameters = {
                            'batch_size': batch_size,
                            'lr': lr,
                            'weight_decay': weight_decay,
                            'patience': patience
                        }
                        print(f"Best Hyperparameters: {best_hyperparameters}")
                        print("Best Accuracy:", best_accuracy)

if __name__ == "__main__":
    main()
