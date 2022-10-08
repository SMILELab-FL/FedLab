import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import sys
import os

from torch import nn

sys.path.append("../../")
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_client import SGDClientTrainer
from fedlab.models import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST

class AsyncTrainer(SGDClientTrainer):
    @property
    def uplink_package(self):
        return [self.model_parameters, self.round]

    def local_process(self, payload, id):
        model_parameters = payload[0]
        self.round = payload[1]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)
        

parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)

parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=100)
args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

model = MLP(784,10)

trainer = AsyncTrainer(model, cuda=args.cuda)
dataset = PathologicalMNIST(root='../../datasets/mnist/', path="../../datasets/mnist/")
if args.rank == 1:
    dataset.preprocess()
trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank)

Manager = ActiveClientManager(trainer=trainer, network=network)
Manager.run()
