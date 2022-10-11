import unittest
import time
from copy import deepcopy

from fedlab.core.client.manager import ClientManager, ActiveClientManager, PassiveClientManager
from fedlab.core.client import ORDINARY_TRAINER, SERIAL_TRAINER
from fedlab.core.network import DistNetwork
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.core.client.trainer import ClientTrainer, SerialClientTrainer
from fedlab.utils import MessageCode, Logger
from fedlab.models import MLP

import torch
import torch.distributed as dist
from torch.multiprocessing import Process



class TestClientTrainer(ClientTrainer):
    def __init__(self, model, cuda=False, device=None, logger=None):
        super().__init__(model, cuda, device)
        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        return [self.model_parameters]

    def local_process(self, payload, id):
        pass 

class TestSerialClientTrainer(SerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False):
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.cache = []

    @property
    def uplink_package(self):
        self.cache = [[self.model_parameters] for _ in range(self.num_clients)]
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list):
        pass


class TestPassiveClientManager(PassiveClientManager):
    def __init__(self, network, trainer, logger=None):
        super().__init__(network, trainer, logger)

    def synchronize(self):
        """Synchronize with server."""
        self._LOGGER.info("Uploading information to server.")
        if self._trainer.type == SERIAL_TRAINER:
            payloads = self._trainer.uplink_package
            for idx, elem in enumerate(payloads):
                self._LOGGER.info("SERIAL_TRAINER trying to synchronize sending client-{idx}'s information...")
                self._network.send(content=elem,
                                message_code=MessageCode.ParameterUpdate,
                                dst=0)
                self._LOGGER.info("SERIAL_TRAINER synchronize client-{idx} done.")
                            
        if self._trainer.type == ORDINARY_TRAINER:
            self._LOGGER.info("ORDINARY_TRAINER trying to synchronize sending...")
            self._network.send(content=self._trainer.uplink_package,
                                message_code=MessageCode.ParameterUpdate,
                                dst=0)
            self._LOGGER.info("ORDINARY_TRAINER synchronize done.")


if __name__ == "__main__":
    host_ip = 'localhost'
    port = '3333'
    model = MLP(784,10)
    client_network = DistNetwork(address=(host_ip, port),
                                 world_size=2,
                                 rank=1)
    trainer = TestClientTrainer(model=model, cuda=False) 
    client_manager = TestPassiveClientManager(client_network, trainer) 
    client_manager._network.init_network_connection()
    print(f"Client Start to synchronize")
    client_manager.synchronize()
    print(f"Client synchronize done")
    # 3. shutdown client network manually
    client_manager._network.close_network_connection()
