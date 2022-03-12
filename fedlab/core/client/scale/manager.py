# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,``
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ...client import ORDINARY_TRAINER, SERIAL_TRAINER
from ...client.manager import ClientPassiveManager

from ....utils.message_code import MessageCode
from ....utils import Logger

class ScaleClientPassiveManager(ClientPassiveManager):
    """Special client manager for :class:`SerialTrainer`.
        
    We modify the communication agreements to create a mapping from client id to process rank.
    Thus, :class:`ScaleClientPassiveManager` is able to simulate multiple clients in sequence.

    Args:
        network (DistNetwork): Distributed network to use.
        trainer (ClientTrainer): Subclass of :class:`ClientTrainer`, providing :meth:`train` and :attr:`model`. For more client simulation with single process, you are supposed to use :class:`SerialTrainer` here.
        logger (Logger): object of :class:`Logger`.
    """
    def __init__(self, network, trainer, logger=Logger()):
        super().__init__(network, trainer, logger)

    def main_loop(self):
        """Actions to perform when receiving new message, including local training."""
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)
            if message_code == MessageCode.Exit:
                break
            elif message_code == MessageCode.ParameterUpdate:
                model_parameters = payload[0]

                sender_rank, message_code, payload = self._network.recv(src=0)

                id_list = payload[0].tolist()

                # check the trainer type
                if self._trainer.type == SERIAL_TRAINER:
                    self.model_parameters_list = self._trainer.train(
                        model_parameters=model_parameters,
                        id_list=id_list,
                        aggregate=False)
                elif self._trainer.type == ORDINARY_TRAINER:
                    self.model_parameters_list = self._trainer.train(
                        model_parameters=model_parameters)
                self.synchronize()
            else:
                raise ValueError("Invalid MessageCode {}. Please see MessageCode Enum".format(message_code))

    def synchronize(self):
        """Synchronize local model with server actively"""
        self._network.send(content=self.model_parameters_list, message_code=MessageCode.ParameterUpdate, dst=0)
