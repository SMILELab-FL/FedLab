import torch
from torch import nn
from torch.utils.data import dataloader
from torch.utils.data.dataset import Dataset
from fedlab.core.client.trainer import ClientTrainer


class CNN_Mnist(nn.Module):
    def __init__(self):
        super(CNN_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = nn.Sequential(nn.Linear(20,10), nn.Linear(10,5))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.L1Loss()

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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





    