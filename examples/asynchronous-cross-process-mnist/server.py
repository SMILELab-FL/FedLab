import os
import sys
import argparse
from torch import nn

sys.path.append("../../")
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_server import AsyncServerHandler
from fedlab.core.server.manager import AsynchronousServerManager
from fedlab.models import MLP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    model = MLP(784,10)
    
    handler = AsyncServerHandler(model, global_round=5)
    handler.setup_optim(0.5)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)
    Manager = AsynchronousServerManager(handler=handler, network=network)

    Manager.run()
