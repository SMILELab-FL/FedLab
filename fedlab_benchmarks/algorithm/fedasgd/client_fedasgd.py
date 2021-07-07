import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import sys
import os

from torch import nn
sys.path.append('/home/zengdun/FedLab/')

from fedlab_utils.logger import logger
from fedlab_core.client.topology import ClientActiveTopology
from fedlab_core.client.trainer import ClientSGDTrainer
from fedlab_utils.dataset.sampler import DistributedSampler
from fedlab_utils.models.lenet import LeNet
from fedlab_core.network import DistNetwork

def get_dataset(args, dataset='MNIST', transform=None, root='/home/zengdun/datasets/mnist/'):
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
    trainset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, sampler=DistributedSampler(trainset, rank=args.local_rank,
                                                                                   num_replicas=args.world_size - 1),
                                              batch_size=128,
                                              drop_last=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             drop_last=False, num_workers=2, shuffle=False)
    return trainloader, testloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--local_rank', type=int)
    
    args = parser.parse_args()
    args.cuda = False

    model = LeNet()
    trainloader, testloader = get_dataset(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    handler = ClientSGDTrainer(model, trainloader, epoch=2, optimizer=optimizer, criterion=criterion, cuda=args.cuda)

    network = DistNetwork(address=(args.server_ip, args.server_port), world_size=args.world_size, rank=args.local_rank)
    topology = ClientActiveTopology(handler=handler, network=network)
    topology.run()
