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

def load_model(weights_path="mnist_cnn.pth"):
    device = torch.device("cpu")
    model = CNN()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model