import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from data_processing import Data_Generator
from modeling import CustomCNN, AlexNet, ResNet, VisionTransformer
import argparse

def train_validate(model, train_loader, val_loader, criterion, optimizer, epochs, threshold, device):
    # Training loop
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)
        mean_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(mean_train_loss)

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            total_test_loss = 0
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_samples += images.size(0)

                total_test_loss += criterion(outputs, labels.view(-1, 1).float()).item()
                percentage_difference = torch.abs((outputs - labels.view(-1, 1).float()) / labels.view(-1, 1).float()) * 100
                total_correct += torch.sum(percentage_difference <= threshold)

            accuracy = total_correct / total_samples
            mean_test_loss = total_test_loss / len(val_loader.dataset)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {mean_train_loss:.4f}, Val Loss: {mean_test_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            if mean_test_loss < best_val_loss:
                best_val_loss = mean_test_loss
                counter = 0
            else:
                counter += 1

            if counter >= 3:  # Early stopping criterion
                break

    return best_val_loss, mean_train_loss, accuracy

def cross_validate(model_type, num_conv_layers, image_folder, num_splits=5, batch_size=64, epochs=40, lr=0.0001, weight_decay=0.001):
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
    
    kfold = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    accuracies = []
    maes = []
    mses = []
    threshold = 5.0

    image_filenames = glob(image_folder)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_filenames), 1):
        train_files = [image_filenames[i] for i in train_idx]
        val_files = [image_filenames[i] for i in val_idx]
        
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Assuming grayscale images
        ])

        train_data_generator = Data_Generator(train_files, transform=data_transform)
        val_data_generator = Data_Generator(val_files, transform=data_transform)

        train_data_loader = DataLoader(train_data_generator, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(val_data_generator, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training and validation
        best_val_loss, mean_train_loss, accuracy = train_validate(model, train_data_loader, val_data_loader, criterion, optimizer, epochs, threshold, device)

        # Store metrics
        accuracies.append(accuracy)
        maes.append(mean_train_loss)
        mses.append(best_val_loss)
        print(f"Fold {fold}: Best Val Loss: {best_val_loss:.4f}, Train Loss: {mean_train_loss:.4f}, Accuracy: {accuracy:.4f}")

    mean_accuracy = sum(accuracies) / len(accuracies)
    mean_mae = sum(maes) / len(maes)
    mean_mse = sum(mses) / len(mses)
    print(f"Mean Accuracy across all folds: {mean_accuracy:.4f}")
    print(f"Mean MAE across all folds: {mean_mae:.4f}")
    print(f"Mean MSE across all folds: {mean_mse:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Perform cross-validation on models for cell counting')
    parser.add_argument('--model', type=str, choices=['CustomCNN', 'AlexNet', 'ResNet', 'VisionTransformer'],
                        help='Select the model type for training')
    parser.add_argument('--num-conv-layers', type=int, default=1, choices=range(1, 5),
                        help='Select the number of convolutional layers for CustomCNN (1-4)')
    parser.add_argument('--image-folder', type=str, help='Path to the image folder')
    parser.add_argument('--k-fold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='Weight decay')

    args = parser.parse_args()

    # Perform cross-validation
    cross_validate(args.model, args.num_conv_layers, args.image_folder, args.k_fold, args.batch_size, args.epochs, args.learning_rate, args.weight_decay)

if __name__ == "__main__":
    main()
