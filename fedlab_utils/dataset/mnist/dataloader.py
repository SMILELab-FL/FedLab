import torchvision
import torchvision.transforms as transforms
import torch

from fedlab_utils.dataset.sampler import DistributedSampler


# def get_dataloader(args, root='/home/zengdun/datasets/mnist/'):
def get_dataloader_mnist(args, root='../../../data/mnist'):
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
        sampler=DistributedSampler(trainset,
                                   rank=args.local_rank,
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