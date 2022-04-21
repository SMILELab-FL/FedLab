import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import sys
import os

from torch import nn

sys.path.append("../../")
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.client.trainer import SGDClientTrainer
from fedlab.utils.dataset.sampler import RawPartitionSampler
from fedlab.core.network import DistNetwork


# torch model
class MLP(nn.Module):

    def __init__(self, input_size=784, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AsyncTrainer(SGDClientTrainer):

    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=False,
                 logger=None):
        super().__init__(model, data_loader, epochs, optimizer, criterion,
                         cuda, logger)
        self.time = 0

    def local_process(self, payload):
        self.time = payload[1].item()
        return super().local_process(payload)

    @property
    def uplink_package(self):
        return [self.model_parameters, torch.Tensor([self.time])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--rank', type=int)

    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()
    args.root = '../../datasets/mnist/'
    args.cuda = True

    # get mnist dataset
    root = "../../tests/data/mnist/"
    trainset = torchvision.datasets.MNIST(root=root,
                                          train=True,
                                          download=True,
                                          transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=RawPartitionSampler(trainset,
                                    client_id=args.rank,
                                    num_replicas=args.world_size - 1),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.world_size)

    model = MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    handler = AsyncTrainer(model,
                           trainloader,
                           epochs=args.epoch,
                           optimizer=optimizer,
                           criterion=criterion,
                           cuda=args.cuda)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank)

    Manager = ActiveClientManager(trainer=handler, network=network)
    Manager.run()
