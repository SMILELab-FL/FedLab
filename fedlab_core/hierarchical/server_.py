import os
import sys

sys.path.append('/home/zengdun/FedLab/')

from fedlab_utils.models.lenet import LeNet
from fedlab_core.server.handler import SyncParameterServerHandler
from fedlab_core.server.topology import ServerSynchronousTopology
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3001')
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()

    model = LeNet().cpu()
    ps = SyncParameterServerHandler(model, client_num_in_total=args.world_size-1)

    topology = ServerSynchronousTopology(handler=ps, server_address=(args.server_ip, args.server_port))

    topology.run()
