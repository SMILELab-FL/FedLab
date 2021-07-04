"""
functions associated with data and dataset operations

"""
from torch.utils.data import Dataset


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