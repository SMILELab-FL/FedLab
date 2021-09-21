# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ...client import ORDINARY_TRAINER, SERIAL_TRAINER
from ...client.manager import ClientPassiveManager

from ...communicator.package import Package
from ...communicator.processor import PackageProcessor

from fedlab.utils.message_code import MessageCode


class ScaleClientPassiveManager(ClientPassiveManager):
    """Special client manager for :class:`SerialTrainer`.
        
    We modify the communication agreements creating mapping between process rank and client id.
    In this way, :class:`Manager` is able to represent multiple clients.

    Args:
        network (DistNetwork): Distributed network to use.
        handler (ClientTrainer): Subclass of :class:`ClientTrainer`, providing :meth:`train` and :attr:`model`.
    """
    def __init__(self, network, trainer):
        super().__init__(network, trainer)

    def on_receive(self, sender_rank, message_code, payload):
        """Actions to perform when receiving new message, including local training

        .. note::
            Customize the control flow of client corresponding with :class:`MessageCode`.

        Args:
            sender_rank (int): Rank of sender
            message_code (MessageCode): Agreements code defined in :class:`MessageCode`
            payload (list[torch.Tensor]): A list of tensors received from sender.
        """
        if message_code == MessageCode.ParameterUpdate:
            model_parameters = payload[0]

            _, message_code, payload = PackageProcessor.recv_package(src=0)
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

    def synchronize(self):
        """Synchronize local model with server actively

        .. note::
            Communication agreements related. Overwrite this function to customize package for synchronizing.
        """
        pack = Package(message_code=MessageCode.ParameterUpdate,
                       content=self.model_parameters_list)
        PackageProcessor.send_package(package=pack, dst=0)
