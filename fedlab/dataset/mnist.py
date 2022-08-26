
import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from .dataset import FedLabDataset, BaseDataset
from ..utils.dataset.functional import noniid_slicing, random_slicing

class PathologicalMNIST(FedLabDataset):
    """_summary_

        Args:
            root (_type_): _description_
            path (_type_): _description_
            num (_type_): _description_
            shards (int, optional): _description_. Defaults to 2.
            download (bool, optional): _description_. Defaults to True.
        """
    def __init__(self, root, path) -> None:
        self.root = os.path.expanduser(root)
        self.path = path 

    def preprocess(self, num, shards, download=True):
        self.num = num
        self.shards = shards
        self.download=download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))
        # train
        mnist = torchvision.datasets.MNIST(self.root, train=True, download=self.download, transform=transforms.ToTensor())
        data_indices = noniid_slicing(mnist, self.num, self.shards)
        
        samples, labels = [], []
        for x, y in mnist:
            samples.append(x)
            labels.append(y)
        for id, indices in data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.path, "train", "data{}.pkl".format(id)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.path, type, "data{}.pkl".format(id)))        
        return dataset

    def get_dataloader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader

class RotatedMNIST(FedLabDataset):
    def __init__(self, root, path, num, download=True) -> None:
        self.root = os.path.expanduser(root)
        self.path = path 
        self.num = num

        
    def pre_process(self, thetas = [0, 90, 180, 270], download=True):
        self.download=download
        # "./datasets/rotated_mnist/"
        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        # train
        mnist = torchvision.datasets.MNIST(self.root, train=True, download=self.download, transform=transforms.ToTensor())
        id = 0
        to_tensor = transforms.ToTensor()
        for theta in thetas:
            rotated_data = []
            labels = []
            partition = random_slicing(mnist, int(self.num/len(thetas)))
            for x, y in mnist:
                x = to_tensor(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
                labels.append(y)
            for key, value in partition.items():
                data = [rotated_data[i] for i in value]
                label = [labels[i] for i in value]
                dataset = BaseDataset(data, label)
                torch.save(dataset, os.path.join(self.dir, "train" ,"data{}.pkl".format(id)))
                id += 1

        # test
        mnist_test = torchvision.datasets.MNIST(self.root, train=False, download=self.download, transform=transforms.ToTensor())
        labels = mnist_test.targets
        for i, theta in enumerate(thetas):
            rotated_data = []
            for x, y in mnist_test:
                x = to_tensor(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            dataset = BaseDataset(rotated_data, labels)
            torch.save(dataset, os.path.join(self.dir,"test", "data{}.pkl".format(i)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader


    