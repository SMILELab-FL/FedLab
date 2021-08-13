import torchvision

if __name__ == "__main__":
    root = './'
    trainset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
    )

    testset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=True,
    )
