import os
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataset_animals import AnimalsDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

def toPILImage(image_path):
    return Image.open(image_path).convert('RGB')

training_data = AnimalsDataset(
    annotations_file=f"{os.getcwd()}/datasets/animals10_train",
    img_dir=f"{os.getcwd()}/datasets/Animals-10",
    split="70/15/15",
    transform=toPILImage
)
test_data = AnimalsDataset(
    annotations_file=f"{os.getcwd()}/datasets/animals10_test",
    img_dir=f"{os.getcwd()}/datasets/Animals-10",
    split="70/15/15",
    transform=toPILImage
)
validation_data = AnimalsDataset(
    annotations_file=f"{os.getcwd()}/datasets/animals10_validation",
    img_dir=f"{os.getcwd()}/datasets/Animals-10",
    split="70/15/15",
    transform=toPILImage
)

img, label = training_data[0]
img.show()
