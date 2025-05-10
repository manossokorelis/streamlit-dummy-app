# model.py

import torch
import torch.nn as nn

# CNN Model (same as training script)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# def load_model(weights_path="mnist_cnn.pth"):
#     device = torch.device("cpu")
#     model = CNN()
#     model.load_state_dict(torch.load(weights_path, map_location=device))
#     model.eval()
#     return model

def load_model():
    # Load a pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify the first conv layer to accept 1-channel input instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replace the final fully connected layer to match MNIST (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    # Load trained weights if available (optional)
    # model.load_state_dict(torch.load("resnet_mnist.pth", map_location="cpu"))
    model.eval()
    return model