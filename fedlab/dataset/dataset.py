from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base dataset iterator"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class FedLabDataset:
    def __init__(self) -> None:
        self.num = None     # the number of dataset indexed from 0 to num-1.
        self.root = None    # the raw dataset.
        self.path = None    # path to save the partitioned datasets.

    def preprocess(self):
        """Define the dataset partition process"""
        raise NotImplementedError()

    def get_dataset(self, id, type="train"):
        """"Get dataset"""
        raise NotImplementedError()

    def get_dataloader(self, id, batch_size, type="train"):
        """Get data loader"""
        raise NotImplementedError()