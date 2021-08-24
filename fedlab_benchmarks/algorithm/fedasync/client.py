import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import sys
import os

from torch import nn

sys.path.append('../../../')

from fedlab.core.client.manager import ClientActiveManager
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.utils.dataset.sampler import FedDistributedSampler
from fedlab.core.network import DistNetwork

from fedlab_benchmarks.models.lenet import LeNet


def get_dataset(args):
    """
    :param dataset_name:
    :param transform:
    :param batch_size:
    :return: iterators for the datasetaccuracy_score
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root=args.root,
                                          train=True,
                                          download=True,
                                          transform=train_transform)
    testset = torchvision.datasets.MNIST(root=args.root,
                                         train=False,
                                         download=True,
                                         transform=test_transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=FedDistributedSampler(trainset,
                                      client_id=args.rank,
                                      num_replicas=args.world_size - 1),
        batch_size=128,
        drop_last=True,
        num_workers=2)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=len(testset),
                                             drop_last=False,
                                             num_workers=2,
                                             shuffle=False)
    return trainloader, testloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--rank', type=int)

    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()
    args.root = '../../../../datasets/mnist/'
    args.cuda = True

    model = LeNet()
    trainloader, testloader = get_dataset(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    handler = ClientSGDTrainer(model,
                               trainloader,
                               epochs=args.epoch,
                               optimizer=optimizer,
                               criterion=criterion,
                               cuda=args.cuda)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank)
    Manager = ClientActiveManager(handler=handler, network=network)
    Manager.run()
