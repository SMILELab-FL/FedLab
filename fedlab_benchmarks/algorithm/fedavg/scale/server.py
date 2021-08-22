from fedlab.core.communicator.processor import PackageProcessor
from fedlab.core.communicator.package import Package
import os
import sys
import argparse

sys.path.append('../../../../')
from fedlab.utils.logger import Logger
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.network_manager import NetworkManager
from fedlab.core.network import DistNetwork
from fedlab.core.coordinator import coordinator
from setting import get_model
import torch
from setting import get_dataset
import torchvision
from torchvision import transforms
from fedlab.utils.functional import evaluate


class ScaleSynchronousServer(NetworkManager):
    def __init__(self, handler, network, logger):
        super().__init__(handler, network, logger=logger)

    def setup(self):
        super().setup()
        map = {}
        for rank in range(1, self._network.world_size):
            _, _, content = PackageProcessor.recv_package(src=rank)
            data = content[0].item()
            map[rank] = data
        self.coordinator = coordinator(map)

    def run(self):
        self.setup()

    def activate_clients(self):
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)
        for key, values in rank_dict:
            pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    model = get_model(args)
    LOGGER = Logger(log_name="server")
    handler = SyncParameterServerHandler(model,
                                         client_num_in_total=args.world_size -
                                         1,
                                         global_round=args.round,
                                         logger=LOGGER,
                                         sample_ratio=args.sample)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0,
                          ethernet=args.ethernet)

    manager_ = ScaleSynchronousServer(handler=handler,
                                      network=network,
                                      logger=LOGGER)
    manager_.run()
