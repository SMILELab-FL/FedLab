import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from .basic_dataset import FedDataset

class CelebADataset(Dataset):
    def __init__(self,
                 client_id: int,
                 client_str: str,
                 data: list,
                 targets: list,
                 image_root: str,
                 transform=None):
        """get `Dataset` for CelebA dataset
         Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): input image name list data
            targets (list):  output label list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.image_root = Path(__file__).parent.resolve() / image_root
        self.transform = transform
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _process_data_target(self):
        """process client's data and target
        """
        data = []
        targets = []
        for idx in range(len(self.data)):
            image_path = self.image_root / self.data[idx]
            image = Image.open(image_path).convert('RGB')
            data.append(image)
            targets.append(torch.tensor(self.targets[idx], dtype=torch.long))
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        target = self.targets[index]
        return data, target