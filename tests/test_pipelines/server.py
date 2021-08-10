import os
import sys
import argparse

sys.path.append('../../')
from fedlab.utils.logger import Logger
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork
from setting import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    model = get_model(args)

    ps = SyncParameterServerHandler(model, client_num_in_total=args.world_size-1)
    network = DistNetwork(address=(args.server_ip, args.server_port), world_size=args.world_size, rank=0)
    Manager = ServerSynchronousManager(handler=ps, network=network)
        
    Manager.run()
