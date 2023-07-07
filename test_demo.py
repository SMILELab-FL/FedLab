from json import load
import os
import argparse
import random
from copy import deepcopy
from munch import Munch

import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu

from fedlab.models.mlp import MLP
from fedlab.models.cnn import CNN_MNIST
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST

from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.fedprox import FedProxServerHandler, FedProxSerialClientTrainer
from fedlab.contrib.algorithm.scaffold import ScaffoldSerialClientTrainer, ScaffoldServerHandler
from fedlab.contrib.algorithm.fednova import FedNovaSerialClientTrainer, FedNovaServerHandler
from fedlab.contrib.algorithm.feddyn import FedDynSerialClientTrainer, FedDynServerHandler

args = Munch()
args.total_client = 100
args.com_round = 10000
args.sample_ratio = 0.2
args.batch_size = 600
args.epochs = 5
args.lr = 0.05

args.preprocess = False
args.seed = 0

args.alg = "fedavg"  # fedavg, fedprox, scaffold, fednova, feddyn
# optim parameter

args.mu = 0.1  # fedprox
args.alpha = 0.01  # feddyn

setup_seed(args.seed)
test_data = torchvision.datasets.MNIST(root="./datasets/mnist/",
                                       train=False,
                                       transform=transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=1024)

model = MLP(784, 10)
# model = CNN_MNIST()

if args.alg == "fedavg":
    handler = SyncServerHandler(model=model,
                                global_round=args.com_round,
                                sample_ratio=args.sample_ratio)
    trainer = SGDSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

if args.alg == "fedprox":
    handler = FedProxServerHandler(model=model,
                                   global_round=args.com_round,
                                   sample_ratio=args.sample_ratio)
    trainer = FedProxSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr, mu=args.mu)

if args.alg == "scaffold":
    handler = ScaffoldServerHandler(model=model,
                                    global_round=args.com_round,
                                    sample_ratio=args.sample_ratio)
    handler.setup_optim(lr=args.lr)

    trainer = ScaffoldSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

if args.alg == "fednova":
    handler = FedNovaServerHandler(model=model,
                                   global_round=args.com_round,
                                   sample_ratio=args.sample_ratio)
    handler.setup_optim()
    trainer = FedNovaSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

if args.alg == "feddyn":
    handler = FedDynServerHandler(model=model,
                                  global_round=args.com_round,
                                  sample_ratio=args.sample_ratio)
    handler.setup_optim(alpha=args.alpha)
    trainer = FedDynSerialClientTrainer(model, args.total_client, cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr, args.alpha)

mnist = PathologicalMNIST(root='./datasets/mnist/', path="./datasets/mnist/pathmnist", num_clients=args.total_client, shards=200, preprocess=True)
# mnist = PartitionedMNIST(root='./datasets/mnist/',
#                          path="./datasets/mnist/fedmnist_iid",
#                          num_clients=args.total_client,
#                          partition="iid",
#                          dir_alpha=args.alpha,
#                          preprocess=args.preprocess,
#                          transform=transforms.Compose(
#                              [transforms.ToPILImage(),
#                               transforms.ToTensor()]))
# mnist.preprocess()
trainer.setup_dataset(mnist)

import time

round = 1
accuracy = []
handler.num_clients = trainer.num_clients
while handler.if_stop is False:
    # server side
    sampled_clients = handler.sample_clients()
    broadcast = handler.downlink_package

    # client side
    trainer.local_process(broadcast, sampled_clients)
    uploads = trainer.uplink_package

    # server side
    for pack in uploads:
        handler.load(pack)

    loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)
    accuracy.append(acc)
    print("Round {}, Test Accuracy: {:.4f}, Max Acc: {:.4f}".format(
        round, acc, max(accuracy)))
    if acc >= 0.97:
        break
    round += 1
torch.save(
    accuracy, "./exp_logs/{}, accuracy_{}_B{}_S{}_R{}_Seed{}_T{}.pkl".format(
        args.alg, "mnist", args.batch_size, args.sample_ratio, args.com_round,
        args.seed, time.strftime("%Y-%m-%d-%H:%M:%S")))
