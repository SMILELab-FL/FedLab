"""
functions associated with data and dataset operations
"""
import warnings

import numpy as np
from torchvision import datasets, transforms
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset


class DistillDataset(Dataset):
    """ Dataset with data logit and label """
    def __init__(self, dataset, logit):
        self.data = dataset.data
        self.targets = dataset.targets
        self.logits = logit
        self.transform = dataset.transform
        if len(dataset) != len(logit):
            raise ValueError("Invalid Logit, length does not match")

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):
        data, label, logit = self.data[idx], self.targets[idx], self.logits[idx]
        if self.transform:
            data = self.transform(data)
        return data, label, logit


class BaseDataset(Dataset):
    """ Basic Dataset Class """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.targets[idx]
        if self.transform:
            data = self.transform(data)
        return data, label
