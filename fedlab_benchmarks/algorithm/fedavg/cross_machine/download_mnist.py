import torchvision
if __name__ == "__main__":
    root = '../../../../../datasets/mnist/'

    trainset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
    )
