import os
import sys
import argparse

from torch import manager_path

sys.path.append('../../../../')
from fedlab.utils.logger import logger
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork
from setting import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--server_ip', type=str)
    parser.add_argument('--server_port', type=str)
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ethernet', type=str)
    args = parser.parse_args()

    os.environ["GLOO_SOCKET_IFNAME"] = args.ethernet

    model = get_model(args)
    LOGGER = logger(log_name="server")

    ps = SyncParameterServerHandler(model, client_num_in_total=args.world_size-1, global_round=args.round, logger=LOGGER)
    network = DistNetwork(address=(args.server_ip, args.server_port), world_size=args.world_size, rank=0)
    manager = ServerSynchronousManager(handler=ps, network=network, logger=LOGGER)
    manager.run()
