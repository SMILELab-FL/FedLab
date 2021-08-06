import torch
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.append('../../')

from fedlab.utils.dataset.sampler import FedDistributedSampler
from fedlab.utils.models.lenet import LeNet


def get_dataset(args):
    root = "../data/mnist/"
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root=root,
                                            train=True,
                                            download=True,
                                            transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=FedDistributedSampler(trainset,
                                        client_id=args.rank,
                                        num_replicas=args.world_size - 1),
        batch_size=128,
        drop_last=True,
        num_workers=2)

    return trainloader, None


def get_model(args):
    model = LeNet()
    return model