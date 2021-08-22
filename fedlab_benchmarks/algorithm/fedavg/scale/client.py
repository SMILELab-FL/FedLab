from logging import log
import torch
import argparse
import sys
import os

sys.path.append("../../../../")

from torch import nn
from fedlab.core.network_manager import NetworkManager
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger

from fedlab.core.communicator.package import Package
from fedlab.core.communicator.processor import PackageProcessor

from setting import get_model, get_dataset


class ScaleClientManager(NetworkManager):
    def __init__(self, handler, network, logger):
        super().__init__(handler, network, logger=logger)

    def setup(self):
        super().setup()
        content = torch.Tensor(self._handler.client_num)
        setup_pack = Package(content=content, data_type=1)
        PackageProcessor.send_package(setup_pack, dst=0)

    def on_receive(self, sender_rank, message_code, payload):
        return super().on_receive(sender_rank, message_code, payload)
