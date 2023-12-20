import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoConfig

class CustomCNN(nn.Module):
    def __init__(self, num_conv_layers=1, num_classes=1):
        super(CustomCNN, self).__init__()
        self.conv_layers = self._create_conv_layers(num_conv_layers)
        self.fc = nn.Linear(512, num_classes)  # Modify num_classes according to your task

    def _create_conv_layers(self, num_conv_layers):
        layers = []
        in_channels = 3  # Assuming input images have 3 channels (RGB)

        # Define convolutional layers
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ]
            in_channels = 64  # Update the number of input channels for the next layer

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.fc(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        # Replace the last fully connected layer
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        in_features = self.resnet.fc.out_features
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        outputs = self.resnet(x)
        regression_outputs = self.regression_head(outputs)
        return regression_outputs

class VisionTransformer(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.vit = AutoModel.from_pretrained("google/vit-base-patch16-224")
        self.config = AutoConfig.from_pretrained("google/vit-base-patch16-224")
        self.regression_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        regression_output = self.regression_head(pooled_output)
        return regression_output