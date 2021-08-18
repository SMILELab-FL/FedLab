import os
import sys

sys.path.append('../../../')

from fedlab.core.network import DistNetwork
from fedlab.core.server.handler import AsyncParameterServerHandler
from fedlab.core.server.manager import ServerAsynchronousManager
from fedlab_benchmarks.models.lenet import LeNet

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    model = LeNet().cpu()
    ps = AsyncParameterServerHandler(model,
                                     client_num_in_total=args.world_size - 1)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)
    Manager = ServerAsynchronousManager(handler=ps, network=network)

    Manager.run()
