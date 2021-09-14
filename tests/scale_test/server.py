import sys
import argparse

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.transforms import Scale

sys.path.append("../../")

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from fedlab.utils.functional import AverageMeter, evaluate

from test_setting import MLP

class TestManager(ScaleSynchronousManager):

    def setup(self):
        print("setup")
        super().setup()
        self.coordinator.switch()
        print(self.coordinator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int, default=4)

    parser.add_argument('--round', type=int, default=3)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=0.5)
    args = parser.parse_args()

    model = MLP()

    handler = SyncParameterServerHandler(model,
                                         global_round=args.round,
                                         sample_ratio=args.sample,
                                         cuda=True)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = TestManager(network=network, handler=handler)
    manager_.run()
