# model.py

import torch
import torch.nn as nn
from torchvision import models

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(1, 32, 3, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64 * 5 * 5, 128),
#             nn.ReLU(),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         return self.model(x)

class MobileNetV2Model(nn.Module):
    def __init__(self):
        super(MobileNetV2Model, self).__init__()
        # Load the pretrained MobileNetV2 model
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        # Modify the final classifier layer for 10 classes (digits 0-9)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 10)

    def forward(self, x):
        return self.model(x)

# def load_model(weights_path="mnist_cnn.pth"):
#     model = CNN()
#     model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
#     model.eval()
#     return model
    
def load_model(weights_path="mnist_mobilenetv2.pth"):
    model = MobileNetV2Model()
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.eval()
    return model
    