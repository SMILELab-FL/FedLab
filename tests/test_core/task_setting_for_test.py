import torch
from torch import nn
from torch.utils.data import IterableDataset, dataloader
from torch.utils.data.dataset import Dataset


model = nn.Sequential(nn.Linear(20,10), nn.Linear(10,5))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.L1Loss()

class unittestDataset(Dataset):
    def __init__(self, data,label) -> None:
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

def unittest_dataset():
    data = []
    for _ in range(10):
        data.append(torch.Tensor(size=(20,)))
    target = []
    for _ in range(10):
        target.append(torch.Tensor(size=(5,)))
    dataset = unittestDataset(data,target)
    return dataset
    
def unittest_dataloader():
    dataset = unittest_dataset()
    loader = dataloader.DataLoader(dataset=dataset, batch_size=len(dataset))
    return loader


    