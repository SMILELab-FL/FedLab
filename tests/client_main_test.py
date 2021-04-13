import torchvision.transforms as transforms
import torchvision
import torch
import argparse
import sys
sys.path.append('/home/zengdun/FedLab')

from fedlab_core.client.topology import ClientSyncTop
from fedlab_core.client.handler import ClientSGDHandler
from fedlab_core.utils.sampler import DistributedSampler
from fedlab_core.models.lenet import LeNet



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--server_ip', type=str)
    parser.add_argument('--server_port', type=str)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()
    args.cuda = True

    model = LeNet()

    trainloader, testloader = get_dataset(args)

    handler = ClientSGDHandler(model, trainloader)
    top = ClientSyncTop(client_handler=handler, server_addr=(
        args.server_ip, args.server_port), world_size=3, rank=args.local_rank)
    top.run()
