# model.py

import torch
import torch.nn as nn
from torchvision import models 

def load_model():
    # Load a pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify the first conv layer to accept 1-channel input instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replace the final fully connected layer to match MNIST (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.eval()
    return model

    