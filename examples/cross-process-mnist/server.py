import argparse
import sys
from torch import nn

sys.path.append("../../")
from fedlab.utils.logger import Logger

from fedlab.models.mlp import MLP

from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.network import DistNetwork


parser = argparse.ArgumentParser(description='FL server example')

parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=str)
parser.add_argument('--world_size', type=int)
parser.add_argument('--ethernet', type=str, default=None)

parser.add_argument('--round', type=int, default=3)
parser.add_argument('--sample', type=float, default=1)

args = parser.parse_args()

model = MLP(784,10)
LOGGER = Logger(log_name="server")
handler = FedAvgServerHandler(model,
                                     global_round=args.round,
                                     logger=LOGGER,
                                     sample_ratio=args.sample)


network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=0,
                      ethernet=args.ethernet)

manager_ = SynchronousServerManager(handler=handler,
                                    network=network,
                                    mode="GLOBAL",
                                    logger=LOGGER)
manager_.run()
