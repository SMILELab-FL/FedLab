import torch
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.append('../../../')

from fedlab.utils.dataset.sampler import FedDistributedSampler

from fedlab_benchmarks.models.cnn import CNN_Mnist, LeNet
from fedlab_benchmarks.models.rnn import RNN_Shakespeare, LSTMModel
from fedlab_benchmarks.datasets.leaf_data_process.dataloader import get_LEAF_dataloader
from fedlab_benchmarks.datasets.leaf_data_process.nlp_utils.dataset_vocab.sample_build_vocab import get_built_vocab


def get_dataloader(dataset, client_id):
    if dataset == 'femnist':
        trainloader, testloader = get_LEAF_dataloader(dataset=dataset,
                                                      client_id=client_id)
    elif dataset == 'shakespeare':
        trainloader, testloader = get_LEAF_dataloader(dataset=dataset,
                                                      client_id=client_id)
    else:
        raise ValueError("Invalid dataset:", dataset)

    return trainloader, testloader


def get_dataset(dataset):
    if dataset == 'mnist':
        root = '../../../datasets/mnist/'
        trainset = torchvision.datasets.MNIST(root=root,
                                              train=True,
                                              download=True,
                                              transform=transforms.ToTensor())

        testset = torchvision.datasets.MNIST(root=root,
                                             train=False,
                                             download=True,
                                             transform=transforms.ToTensor())

    elif dataset == 'cifar10':
        root = '../../../datasets/cifar10/'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.MNIST(root=root,
                                              train=True,
                                              download=True,
                                              transform=transform_train)

        testset = torchvision.datasets.MNIST(root=root,
                                             train=False,
                                             download=True,
                                             transform=transform_test)

    else:
        raise ValueError("Invalid dataset:", dataset)

    return trainset, testset


def get_model(dataset):
    if dataset == "mnist":
        model = CNN_Mnist()
    elif dataset == "cifar10":
        model = torchvision.models.resnet18()
    elif dataset == 'femnist':
        model = LeNet(out_dim=62)
    elif dataset == 'shakespeare':
        model = RNN_Shakespeare()
    else:
        raise ValueError("Invalid dataset:", dataset)
    return model