# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import io

# # Define the CNN model architecture (same as during training)
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

# # Function to load the pre-trained model
# def load_model():
#     model = CNN()
#     model.load_state_dict(torch.load("mnist_cnn.pth"))
#     model.eval()
#     return model

# # Function to make predictions
# def predict_digit(model, image_tensor):
#     with torch.no_grad():
#         output = model(image_tensor)
#         _, predicted_class = torch.max(output, 1)
#         return predicted_class.item()