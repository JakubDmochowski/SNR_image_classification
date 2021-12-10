import os
import torch
from torchvision import datasets
from dataset_animals import AnimalsDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

image_size = (224,224)
transform = T.Compose([T.ToPILImage(), T.Resize(image_size), T.PILToTensor()])

training_data = AnimalsDataset(
    annotations_file=f"{os.getcwd()}/datasets/animals10_train",
    img_dir=f"{os.getcwd()}/datasets/Animals-10",
    split="70/15/15",
    transform=transform
)
test_data = AnimalsDataset(
    annotations_file=f"{os.getcwd()}/datasets/animals10_test",
    img_dir=f"{os.getcwd()}/datasets/Animals-10",
    split="70/15/15",
    transform=transform
)
validation_data = AnimalsDataset(
    annotations_file=f"{os.getcwd()}/datasets/animals10_validation",
    img_dir=f"{os.getcwd()}/datasets/Animals-10",
    split="70/15/15",
    transform=transform
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")