import csv
import os
import pickle

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset_animals import AnimalsDataset

classes = dict()


def set_classes():
    global classes
    with open(f"{os.getcwd()}/datasets/animals10_utils", 'rb') as f:
        data = pickle.load(f)
    classes = data


def prepare_data():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    training_data = AnimalsDataset(
        annotations_file=f"{os.getcwd()}/datasets/animals10_train",
        img_dir=f"{os.getcwd()}/datasets/Animals-10",
        transform=train_transform
    )
    test_data = AnimalsDataset(
        annotations_file=f"{os.getcwd()}/datasets/animals10_test",
        img_dir=f"{os.getcwd()}/datasets/Animals-10",
        transform=test_transform
    )
    validation_data = AnimalsDataset(
        annotations_file=f"{os.getcwd()}/datasets/animals10_validation",
        img_dir=f"{os.getcwd()}/datasets/Animals-10",
        transform=test_transform
    )

    train_data_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    validation_data_loader = DataLoader(validation_data, batch_size=64, shuffle=True)

    return train_data_loader, validation_data_loader, test_data_loader


def get_all_data():
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_data = AnimalsDataset(
        annotations_file=f"{os.getcwd()}/datasets/animals10_test",
        img_dir=f"{os.getcwd()}/datasets/Animals-10",
        transform=test_transform
    )

    return DataLoader(all_data, batch_size=1, shuffle=False)


def predict_image(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confusion_matrix = torch.zeros(len(classes), len(classes))
    corrects = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    acc = corrects.double() / len(data_loader.dataset)
    print("Test dataset Accuracy:", acc.cpu().detach().numpy())
    print("Test dataset Confusion_matrix \n", confusion_matrix.cpu().detach().numpy())
def predict_image_svm(model, svm, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confusion_matrix = torch.zeros(len(classes), len(classes))
    corrects = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model_outputs = model(inputs)
        model_outputs = model_outputs.cpu()
        labels = labels.to("cpu")
        preds = svm.predict(model_outputs)
        preds = torch.from_numpy(preds)
        corrects += torch.sum(preds == labels.data)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    acc = corrects.double() / len(data_loader.dataset)
    print("Test dataset Accuracy:", acc.cpu().detach().numpy())
    print("Test dataset Confusion_matrix \n", confusion_matrix.cpu().detach().numpy())

def save_wrong_predicted(wrong_predicted_ids):
    file = open(f"{os.getcwd()}/datasets/wrong_predicted", 'w', encoding='UTF8', newline='')
    writer = csv.writer(file)
    for id in enumerate(wrong_predicted_ids):
        writer.writerow(id)

    file.close()
