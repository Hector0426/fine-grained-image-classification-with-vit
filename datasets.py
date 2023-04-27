import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np


class DataSet(Dataset):
    def __init__(self, file_path, root, transform=None):
        self.df = pd.read_csv(file_path)
        self.transform = transform
        self.root = root
        self.labels = []
        for i in range(len(self.df)):
            self.labels.append(int(self.df.iloc[i, 2]))
        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels).long()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.iloc[idx, 1]
        img = Image.open(os.path.join(self.root, name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.df.iloc[idx, 2]
        return img, label - 1  # in order that the index starts at 0


def read_data_set(file, img_root, batch_size, transform=None, drop_last=True):
    data_set = DataSet(file, img_root, transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    return data_loader
