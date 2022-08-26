from json import load
import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
import sys
import torch

sys.path.append("../../")
torch.manual_seed(0)

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu

from fedlab.models.mlp import MLP
from fedlab.contrib.servers.server import SyncServerHandler
from fedlab.contrib.clients.client import SGDSerialTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.dataset.mnist import PathologicalMNIST

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int)

parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float, default=0.02)

args = parser.parse_args()

model =MLP(784, 10)

handler = SyncServerHandler(model, args.com_round, args.sample_ratio)

trainer = SGDSerialTrainer(model, args.total_client, cuda=True)

dataset = PathologicalMNIST(root='../../tests/data/mnist/', path="../../tests/data/mnist/", num=args.total_client)
#dataset.preprocess()
trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

pipeline = StandalonePipeline(handler, trainer)
pipeline.main()
