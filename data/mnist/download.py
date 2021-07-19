import torchvision
import torchvision.transforms as transforms
import torch


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='./',
                                          train=True,
                                          download=True,
                                          transform=train_transform)
    testset = torchvision.datasets.MNIST(root='./',
                                         train=False,
                                         download=True,
                                         transform=test_transform)