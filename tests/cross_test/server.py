import argparse
import sys

sys.path.append("../../")
from fedlab.utils.logger import Logger
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork
from test_setting import MLP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int, default=3)

    parser.add_argument('--round', type=int, default=3)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    model = MLP()

    LOGGER = Logger(log_name="server")
    handler = SyncParameterServerHandler(model,
                                         global_round=args.round,
                                         logger=LOGGER,
                                         sample_ratio=args.sample)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0,
                          ethernet=args.ethernet)

    manager_ = ServerSynchronousManager(handler=handler,
                                        network=network,
                                        logger=LOGGER)
    manager_.run()
