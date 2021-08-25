import sys
import argparse

import torch

sys.path.append('../../../../')

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.core.coordinator import Coordinator
from fedlab.core.communicator.processor import PackageProcessor
from fedlab.core.communicator.package import Package

from fedlab.utils.functional import evaluate
from fedlab.utils.logger import Logger
from fedlab.utils.message_code import MessageCode

from setting import get_model


class ScaleSynchronousServer(ServerSynchronousManager):
    def __init__(self, network, handler):
        super().__init__(network=network, handler=handler)

    def setup(self):
        super().setup()

        map = {}
        for rank in range(1, self._network.world_size):
            _, _, content = PackageProcessor.recv_package(src=rank)
            map[rank] = content[0].item()

        self.coordinator = Coordinator(map)
        self._handler.client_num_in_total = int(
            sum(self.coordinator.map.values()))

    def activate_clients(self):
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        print(len(clients_this_round))
        print("client id :", clients_this_round)

        for rank, values in rank_dict.items():
            print(rank, values)
            param_pack = Package(message_code=MessageCode.ParameterUpdate,
                                 content=self._handler.model_parameters)
            PackageProcessor.send_package(package=param_pack, dst=rank)

            id_lis = torch.Tensor(values).int()
            act_pack = Package(message_code=MessageCode.ParameterUpdate,
                               content=id_lis,
                               data_type=1)
            PackageProcessor.send_package(package=act_pack, dst=rank)

    def on_receive(self, sender, message_code, payload):
        if message_code == MessageCode.ParameterUpdate:
            for model_parameters in payload:
                update_flag = self._handler.add_model(sender, model_parameters)
                if update_flag is True:
                    return update_flag
        else:
            raise Exception("Unexpected message code {}".format(message_code))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=2)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    model = get_model(args)

    LOGGER = Logger(log_name="server")

    handler = SyncParameterServerHandler(model,
                                         client_num_in_total=1,
                                         global_round=args.round,
                                         logger=LOGGER,
                                         sample_ratio=args.sample)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousServer(network=network, handler=handler)
    manager_.run()
