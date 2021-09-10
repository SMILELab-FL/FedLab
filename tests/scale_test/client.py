import torch
import argparse
import sys
import os

import torchvision
import torchvision.transforms as transforms

sys.path.append("../../../")

from fedlab.core.client.scale.trainer import SerialTrainer
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork

from fedlab.utils.dataset import slicing
from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict

from test_setting import MLP

class TestSHandler(SerialTrainer):
    def __init__(self, model, client_num, aggregator, cuda, logger):
        super().__init__(model,
                         client_num,
                         aggregator,
                         cuda=cuda,
                         logger=logger)

    def _get_dataloader(self, client_id):
        return super()._get_dataloader(client_id)

    def _train_alone(self, model_parameters, train_loader):
        return super()._train_alone(model_parameters, train_loader)


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
                           aggregator=aggregator)

    manager_ = ScaleClientPassiveManager(handler=trainer, network=network)

    manager_.run()