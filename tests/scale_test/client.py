import torch
import argparse
import sys
import os

import torchvision
import torchvision.transforms as transforms

sys.path.append("../../")

from fedlab.core.client.scale.trainer import SerialTrainer
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork

from fedlab.utils.dataset import slicing
from fedlab.utils.dataset.sampler import SubsetSampler
from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict
from fedlab.utils.serialization import SerializationTool

from test_setting import MLP


class TestSHandler(SerialTrainer):
    def __init__(self,
                 model,
                 client_num,
                 aggregator,
                 dataset,
                 data_indices,
                 cuda=True,
                 logger=None):
        super().__init__(model,
                         client_num,
                         aggregator,
                         cuda=cuda,
                         logger=logger)

        self.data_indices = data_indices
        self.dataset = dataset

    def _get_dataloader(self, client_id):
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_indices[client_id],
                                  shuffle=True),
            batch_size=100)
        return train_loader

    def _train_alone(self, model_parameters, train_loader):
        epochs, lr = 2, 0.01
        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._model.train()

        for _ in range(epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                output = self.model(data)

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model_parameters


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3002")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)

    parser.add_argument("--client_num", type=int)

    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str, default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    trainset = torchvision.datasets.MNIST(root='../data/mnist/',
                                          train=True,
                                          download=True,
                                          transform=transforms.ToTensor())

    data_indices = slicing.random_slicing(trainset, num_clients=30)

    model = MLP()

    aggregator = Aggregators.fedavg_aggregate

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)

    trainer = TestSHandler(model=model,
                           client_num=args.client_num,
                           aggregator=aggregator,
                           data_indices=data_indices,
                           dataset=trainset)

    manager_ = ScaleClientPassiveManager(trainer=trainer, network=network)

    manager_.run()