from logging import log
import torch
import argparse
import sys
import os

import torchvision
import torchvision.transforms as transforms

sys.path.append("../../")
from torch import nn
from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from fedlab.utils.dataset import slicing
from fedlab.utils.dataset.sampler import SubsetSampler

from test_setting import MLP

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3002")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str, default=None)
    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.cuda = False

    model = MLP()

    trainset = torchvision.datasets.MNIST(root='../data/mnist/',
                                          train=True,
                                          download=True,
                                          transform=transforms.ToTensor())

    data_indices = slicing.random_slicing(trainset, num_clients=30)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              sampler=SubsetSampler(
                                                  indices=data_indices[0],
                                                  shuffle=True),
                                              batch_size=100)

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

    manager_ = ScaleClientPassiveManager(trainer=trainer, network=network)
    manager_.run()
