import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import sys
import os

sys.path.append('/home/zengdun/FedLab/')

from fedlab_utils.logger import logger
from fedlab_core.client.topology import ClientPassiveTopology
from fedlab_core.client.handler import ClientSGDHandler
from fedlab_utils.models.lenet import LeNet


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
    trainset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128)
    return trainloader
                            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3002')
    parser.add_argument('--local_rank', type=int, default=1)
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()
    args.cuda = False

    model = LeNet()
    trainloader = get_dataset(args)

    handler = ClientSGDHandler(model, trainloader, local_epoch=2, cuda=args.cuda)

    topology = ClientPassiveTopology(handler=handler, server_addr=(args.server_ip, args.server_port), world_size=args.world_size, rank=args.local_rank)
    topology.run()
