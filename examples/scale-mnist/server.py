import sys
import argparse
sys.path.append("../../")

import torch
from torch import nn
import torchvision
from torchvision import transforms

from fedlab.core.server.manager import SynchronousServerManager
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.core.network import DistNetwork
from fedlab.models.mlp import MLP
from fedlab.utils.logger import Logger
from fedlab.utils.functional import AverageMeter, evaluate



parser = argparse.ArgumentParser(description='FL server example')

parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=str, default="3002")
parser.add_argument('--world_size', type=int)
parser.add_argument('--ethernet', type=str, default=None)

parser.add_argument('--round', type=int)
parser.add_argument('--sample', type=float, default=1)

args = parser.parse_args()

model = MLP(784,10)

handler = SyncServerHandler(model,
                            global_round=args.round,
                            sample_ratio=args.sample,
                            cuda=True)

network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=0)

manager_ = SynchronousServerManager(network=network, handler=handler, mode="GLOBAL")

manager_.run()
