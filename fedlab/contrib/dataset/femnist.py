import os
import torch
from torch.utils.data import Dataset


class FemnistDataset(Dataset):
    def __init__(self, client_id: int, client_str: str, data: list,
                 targets: list):
        """get `Dataset` for femnist dataset
         Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): image data list
            targets (list): image class target list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = torch.tensor(self.data,
                                 dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]