import os
import pandas as pd
from torch.utils.data import Dataset


class AnimalsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, split="70/15/15", transform=None, target_transform=None):
        s = split.split('/')
        self.train_p = s[0]
        self.test_p = s[1]
        self.validation_p = s[2]
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = img_path
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label