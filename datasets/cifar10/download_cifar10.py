import torchvision

if __name__ == "__main__":
    root = './'
    trainset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
    )
