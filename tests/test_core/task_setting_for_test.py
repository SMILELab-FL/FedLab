import torch
from torch import nn
from torch.utils.data import IterableDataset, dataloader
from torch.utils.data.dataset import Dataset
from fedlab.core.client.trainer import ClientTrainer

model = nn.Sequential(nn.Linear(20,10), nn.Linear(10,5))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.L1Loss()



class TestTrainer(ClientTrainer):
    def __init__(self, model, cuda):
        super().__init__(model, cuda)

    def train(self, model_parameters):
        pass

class unittestDataset(Dataset):
    def __init__(self, data,label) -> None:
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

def unittest_reg_dataset():
    data = []
    for _ in range(10):
        data.append(torch.Tensor(size=(20,)))
    target = []
    for _ in range(10):
        target.append(torch.Tensor(size=(5,)))
    dataset = unittestDataset(data,target)
    return dataset

def unittest_cls_dataset():
    data = []
    for _ in range(10):
        data.append(torch.Tensor(size=(20,)))
    target = [int(x.argmax().item()) for x in data]
    dataset = unittestDataset(data,target)
    return dataset

def unittest_dataloader(type="reg"):
    if type=="reg":
        dataset = unittest_reg_dataset()
    if type=="cls":
        dataset = unittest_cls_dataset()
    loader = dataloader.DataLoader(dataset=dataset, batch_size=len(dataset))
    return loader





    