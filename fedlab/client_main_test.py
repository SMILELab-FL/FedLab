import argparse
import os
import time

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


def get_dataset(args, transform=None):
    """
    :param dataset_name:
    :param transform:
    :param batch_size:
    :return: iterators for the datasetaccuracy_score
    """
    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)
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
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, sampler=DistributedSampler(trainset),
                                              batch_size=args.batch_size,
                                              drop_last=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             drop_last=False, num_workers=2, shuffle=False)
    return trainloader, testloader

if __name__ == "__main__":
    """
    dist.init_process_group('gloo', init_method='tcp://{}:{}'
                            .format(args.server_ip, args.server_port),
                            rank=args.local_rank, world_size=args.world_size)
    """

    model = LeNet()

    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1,
                        metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='N', help='how often to pull params (default: 5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how often to evaluate and print out')

    parser.add_argument('--server_ip', type=str, default="127.0.0.1")
    parser.add_argument('--server_port', type=int, default=3001)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--world_size', type=int, default=3)
    args = parser.parse_args()
    args.cuda = True
    worker(args, model)
