import os
import sys

sys.path.append('../../../')
# sys.path.append('/home/zengdun/FedLab/')

from fedlab_utils.logger import logger
from models.lenet import LeNet
from fedlab_core.server.handler import AsyncParameterServerHandler
from fedlab_core.server.topology import ServerAsynchronousTopology
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3002')
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()

    model = LeNet().cpu()
    handler_logger = logger(os.path.join("log", "server_handler.txt"), "server")
    ps = AsyncParameterServerHandler(model, client_num_in_total=args.world_size-1, logger=handler_logger)

    topology_logger = logger(os.path.join("log", "server_topology.txt"), "server")
    topology = ServerAsynchronousTopology(handler=ps, server_address=(
        args.server_ip, args.server_port), logger=topology_logger)
        
    topology.run()
