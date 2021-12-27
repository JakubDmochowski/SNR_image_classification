import os


import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset_animals import AnimalsDataset
import torchvision.models as models
from pretrainedAlex import train_alex
import torch

# image_size = (224, 224)
# transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(image_size), transforms.PILToTensor()])
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(244),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    ])

training_data = AnimalsDataset(
    annotations_file=f"{os.getcwd()}/datasets/animals10_train",
    img_dir=f"{os.getcwd()}/datasets/Animals-10",
    transform=transform
)
test_data = AnimalsDataset(
    annotations_file=f"{os.getcwd()}/datasets/animals10_test",
    img_dir=f"{os.getcwd()}/datasets/Animals-10",
    transform=transform
)
validation_data = AnimalsDataset(
    annotations_file=f"{os.getcwd()}/datasets/animals10_validation",
    img_dir=f"{os.getcwd()}/datasets/Animals-10",
    transform=transform
)

train_data_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)
validation_data_loader = DataLoader(validation_data, batch_size=64, shuffle=True)

model = models.resnet18(pretrained=True, progress=True)
print(model)
# train_alex(train_data_loader, test_data_loader)

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
