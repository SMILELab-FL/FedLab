import torch
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('../../../')

from fedlab_utils.dataset.sampler import FedDistributedSampler
from fedlab_utils.models.lenet import LeNet
from fedlab_utils.models.rnn import RNN_Shakespeare
from fedlab_benchmarks.datasets.leaf_data_process.dataloader import get_LEAF_dataloader

def get_dataset(args):
    if args.dataset == 'mnist':
        root = '../../../../datasets/mnist/'
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.MNIST(root=root,
                                              train=True,
                                              download=True,
                                              transform=train_transform)
        testset = torchvision.datasets.MNIST(root=root,
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
    elif args.dataset == 'cifar10':
        pass
    elif args.dataset == 'femnist':
        trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset, client_id=args.rank)
    elif args.dataset == 'shakespeare':
        trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset, client_id=args.rank)
    elif args.dataset == 'sent140':
        trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset, client_id=args.rank)
    else:
        raise ValueError("Invalid dataset:", args.dataset)

    return trainloader, testloader

def get_model(args):
    if args.dataset == "mnist":
        model = LeNet()
    elif args.dataset == 'cifar10':
        pass
    elif args.dataset == 'femnist':
        model = LeNet(out_dim=62)
    elif args.dataset == 'shakespeare':
        model = RNN_Shakespeare()
    elif args.dataset == 'sent140':
        pass
    else:
        raise ValueError("Invalid dataset:", args.dataset)
    return model