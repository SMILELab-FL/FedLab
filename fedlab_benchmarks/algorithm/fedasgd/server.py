import os
import sys


sys.path.append('../../../')

from fedlab_core.network import DistNetwork
from fedlab_utils.models.lenet import LeNet
from fedlab_core.server.handler import AsyncParameterServerHandler
from fedlab_core.server.manager import ServerAsynchronousManager
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    model = LeNet().cpu()
    ps = AsyncParameterServerHandler(model, client_num_in_total=args.world_size-1)

    network = DistNetwork(address=(args.server_ip, args.server_port), world_size = args.world_size, rank=0)
    Manager = ServerAsynchronousManager(handler=ps, network=network)
        
    Manager.run()
