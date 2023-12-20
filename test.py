import torch
import argparse
from data_processing import Data_Generator, load_dataset
from torch.utils.data import DataLoader

def main():   
    parser = argparse.ArgumentParser(description='Test a trained model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image-folder', type=str, help='Path to the folder containing total images')

    args = parser.parse_args()
    
    # Load the model
    model = torch.load(args.model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    _, _, test_data_loader = load_dataset(args.image_folder, 0.8, 1)

    threshold = 5.0
    model.eval()

    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        total_test_loss = 0
        test_loss_mae = 0
        test_loss_mse = 0
        num_batch = 0

        for images, labels, _ in test_data_loader:
            num_batch += 1
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            total_samples += images.size(0)

            test_loss_mae += torch.abs(outputs - labels.view(-1, 1).float()).sum().item()
            test_loss_mse += criterion(outputs, labels.view(-1, 1).float()).item() * images.size(0)
            percentage_difference = torch.abs((outputs - labels.view(-1, 1).float()) / labels.view(-1, 1).float()) * 100
            total_correct += torch.sum(percentage_difference <= threshold)

        mean_mae = test_loss_mae / total_samples
        mean_mse = test_loss_mse / len(test_data_loader.dataset)
        accuracy = total_correct / total_samples

        print(f"Test Loss MAE: {mean_mae:.4f}")
        print(f"Test Loss MSE: {mean_mse:.4f}")
        print(f"Test ACP: {accuracy:.4f}")

if __name__ == "__main__":
    main()
