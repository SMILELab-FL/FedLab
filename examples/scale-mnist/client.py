import torch
import argparse
import sys
sys.path.append("../../")
import os
from torch import nn

import torchvision
import torchvision.transforms as transforms

from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork

from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict


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


parser = argparse.ArgumentParser(description="Distbelief training example")

parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=str, default="3002")
parser.add_argument("--world_size", type=int)
parser.add_argument("--rank", type=int)
parser.add_argument("--ethernet", type=str, default=None)

args = parser.parse_args()

trainset = torchvision.datasets.MNIST(root='../../tests/data/mnist/',
                                      train=True,
                                      download=True,
                                      transform=transforms.ToTensor())

data_indices = load_dict("mnist_partition.pkl")

# Process rank x represent client id from (x-1)*10 - (x-1)*10 +10
# e.g. rank 5 <--> client 40-50
client_id_list = [
    i for i in range((args.rank - 1) * 10, (args.rank - 1) * 10 + 10)
]

# get corresponding data partition indices
sub_data_indices = {
    idx: data_indices[cid]
    for idx, cid in enumerate(client_id_list)
}

model = MLP()

aggregator = Aggregators.fedavg_aggregate

network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank,
                      ethernet=args.ethernet)

trainer = SubsetSerialTrainer(model=model,
                              dataset=trainset,
                              data_slices=sub_data_indices,
                              aggregator=aggregator,
                              args={
                                  "batch_size": 100,
                                  "lr": 0.02,
                                  "epochs": 5
                              })

manager_ = ScaleClientPassiveManager(trainer=trainer, network=network)

manager_.run()