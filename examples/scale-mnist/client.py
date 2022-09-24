import torch
import argparse
import sys
sys.path.append("../../")
import os
from torch import nn

import torchvision
import torchvision.transforms as transforms

from fedlab.contrib.clients import SGDSerialClientTrainer
from fedlab.core.client import PassiveClientManager
from fedlab.core.network import DistNetwork

from fedlab.dataset.pathological_mnist import PathologicalMNIST
from fedlab.models.mlp import MLP
from fedlab.utils.logger import Logger

parser = argparse.ArgumentParser(description="Distbelief training example")

parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=str, default="3002")
parser.add_argument("--world_size", type=int)
parser.add_argument("--rank", type=int)
parser.add_argument("--ethernet", type=str, default=None)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=100)

args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False


model = MLP(784, 10)

network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank,
                      ethernet=args.ethernet)

trainer = SGDSerialClientTrainer(model, 10, cuda=args.cuda)

dataset = PathologicalMNIST(root='../../tests/data/mnist/',
                            path="../../tests/data/mnist/",
                            num=100)

trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

manager_ = PassiveClientManager(trainer=trainer, network=network)
manager_.run()
