import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image

class Data_Generator(Dataset):
    def __init__(self, image_filenames, transform=None):
        self.image_filenames = image_filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx])

        if self.transform:
            image = self.transform(image)

        ground_truth_file = self.image_filenames[idx].replace('images', 'ground_truth').replace('.tiff', '.csv')
        labels = pd.read_csv(ground_truth_file)
        num_labels = len(labels)

        #return torch.tensor(image, dtype=torch.float32), num_labels, filename
        return image, num_labels, self.image_filenames[idx]

def load_dataset(model=None, image_path, train_test_ratio=0.8, batch_size=32, curriculum=None):
    # Get paths for images
    image_filenames = glob(image_path)

    # split toatl files randomly into train, validation and test
    train_files, rest_files = train_test_split(image_filenames, train_test_ratio, random_state=100)
    val_files, test_files = train_test_split(rest_files, 0.5, random_state=100)
   
    torch.manual_seed(42)

    # Define transformations for preprocessing
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Assuming grayscale images
    ])

    train_data_generator = Data_Generator(train_files, transform=data_transform)
    val_data_generator = Data_Generator(val_files, transform=data_transform)
    test_data_generator = Data_Generator(test_files, transform=data_transform)

    train_data_loader = DataLoader(train_data_generator, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data_generator, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data_generator, batch_size=1, shuffle=True)
    
    if curriculum:
        # Calculate losses on the entire training set and sort based on loss
        model.eval()
        all_losses = []
        for images, labels, _ in train_data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = torch.nn.MSELoss(outputs, labels.view(-1, 1).float())
            loss_value = loss.item()  # Get the scalar value from the tensor
            all_losses.append(loss_value)
            
        sorted_train_files = [x for _, x in sorted(zip(all_losses, train_files))]

        train_data_generator = Data_Generator(sorted_train_files, transform=data_transform)

        train_data_loader = DataLoader(train_data_generator, batch_size=batch_size)

    return train_data_loader, val_data_loader, test_data_loader