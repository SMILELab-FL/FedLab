import sys
import argparse

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.transforms import Scale

sys.path.append("../../../")

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from fedlab.utils.functional import AverageMeter, evaluate

from test_setting import MLP

# python server.py --world_size 11
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=100)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=0.1)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    model = MLP()

    testset = torchvision.datasets.MNIST(root='../data/mnist/',
                                         train=False,
                                         download=True,
                                         transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=int(len(testset) / 10),
                                             drop_last=False,
                                             shuffle=False)

    handler = SyncParameterServerHandler(model,
                                         global_round=args.round,
                                         sample_ratio=args.sample,
                                         test_loader=testloader,
                                         cuda=True,
                                         args=args)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler)
    manager_.run()
