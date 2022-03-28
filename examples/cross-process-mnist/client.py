
import argparse
import sys

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

sys.path.append("../../")

from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from fedlab.utils.dataset.sampler import RawPartitionSampler

parser = argparse.ArgumentParser(description="Distbelief training example")

parser.add_argument("--ip", type=str)
parser.add_argument("--port", type=str)
parser.add_argument("--world_size", type=int)
parser.add_argument("--rank", type=int)
parser.add_argument("--ethernet", type=str, default=None)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--cuda", type=bool, default=False)
args = parser.parse_args()

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

model = MLP()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

network = DistNetwork(
    address=(args.ip, args.port),
    world_size=args.world_size,
    rank=args.rank,
    ethernet=args.ethernet,
)

LOGGER = Logger(log_name="client " + str(args.rank))

trainer = ClientSGDTrainer(
    model,
    trainloader,
    epochs=args.epoch,
    optimizer=optimizer,
    criterion=criterion,
    cuda=args.cuda,
    logger=LOGGER,
)

manager_ = ClientPassiveManager(trainer=trainer,
                                network=network,
                                logger=LOGGER)
manager_.run()
