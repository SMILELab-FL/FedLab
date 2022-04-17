import argparse
import sys

sys.path.append("../../")

from fedlab.core.network import DistNetwork
from fedlab.core.server.hierarchical import Scheduler

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='FedLab scheduler (middle server) example')

    # server connector
    parser.add_argument('--ip_u', type=str, default="127.0.0.1")
    parser.add_argument('--port_u', type=str, default="3002")
    parser.add_argument('--world_size_u', type=int, default=2)
    parser.add_argument('--rank_u', type=int, default=1)
    parser.add_argument('--ethernet_u', type=str, default=None)

    # client connector
    parser.add_argument('--ip_l', type=str, default="127.0.0.1")
    parser.add_argument('--port_l', type=str, default="3001")
    parser.add_argument('--world_size_l', type=int, default=3)
    parser.add_argument('--rank_l', type=int, default=0)
    parser.add_argument('--ethernet_l', type=str, default=None)

    args = parser.parse_args()

    network_upper = DistNetwork(address=(args.ip_u, args.port_u),
                                world_size=args.world_size_u,
                                rank=args.rank_u,
                                ethernet=args.ethernet_u)
    network_lower = DistNetwork(address=(args.ip_l, args.port_l),
                                world_size=args.world_size_l,
                                rank=args.rank_l,
                                ethernet=args.ethernet_l)

    middle_server = Scheduler(network_upper, network_lower)

    middle_server.run()
