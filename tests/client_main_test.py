import argparse
import os
import time
import sys
sys.path.append('/home/zengdun/FedLab')

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

from fedlab_core.models.lenet import LeNet
from fedlab_core.utils.messaging import recv_message, send_message, MessageCode
from fedlab_core.utils.serialization import ravel_model_params, unravel_model_params
from fedlab_core.utils.sampler import DistributedSampler
from fedlab_core.client.handler import ClientSGDHandler
from fedlab_core.client.topology import ClientSyncTop


def get_dataset(args, dataset='MNIST', transform=None, root='../../datasets/mnist/'):
    """
    :param dataset_name:
    :param transform:
    :param batch_size:
    :return: iterators for the datasetaccuracy_score
    """
    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)

    if dataset == 'MNIST':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(
            root=root, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(
            root=root, train=False, download=True, transform=test_transform)
    else:
        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ])

        test_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, sampler=DistributedSampler(trainset, rank=args.local_rank, num_replicas=args.world_size-1),
                                              batch_size=128,
                                              drop_last=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             drop_last=False, num_workers=2, shuffle=False)
    return trainloader, testloader


def module_test(topology, backend_worker):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1,
                        metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='N', help='how often to pull params (default: 5)')

    parser.add_argument('--server_ip', type=str, default="127.0.0.1")
    parser.add_argument('--server_port', type=int, default=3001)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--world_size', type=int, default=3)
    args = parser.parse_args()
    args.cuda = True

    model = LeNet()

    trainloader, testloader = get_dataset(args)

    handler = ClientSGDHandler(model, trainloader)
    top = ClientSyncTop(backend_handler=handler, server_addr=(
        '127.0.0.1', '3001'), world_size=3, rank=args.local_rank)
    top.run()
