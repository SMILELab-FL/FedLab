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

import torch
import threading

from ...server.manager import ServerSynchronousManager
from ...communicator.processor import PackageProcessor
from ...communicator.package import Package
from ....utils.message_code import MessageCode
from ....utils import Logger

class ScaleSynchronousManager(ServerSynchronousManager):
    """ServerManager used in scale scenario."""
    def __init__(self, network, handler, logger=Logger()):
        super().__init__(network, handler, logger)

    def activate_clients(self):
        """Use client id mapping: Coordinator. 

        Here we use coordinator to find the rank client process with specific client_id.
        """
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client Activation Procedure")
        for rank, values in rank_dict.items():
            self._LOGGER.info("rank {}, client ids {}".format(rank, values))
            # Send parameters
            self._network.send(content=self._handler.model_parameters, message_code=MessageCode.ParameterUpdate, dst=rank)
            # Send activate id list
            id_list = torch.Tensor(values).to(torch.int32)
            self._network.send(content=id_list, message_code=MessageCode.ParameterUpdate, dst=rank)

    def main_loop(self):
        while self._handler.stop_condition() is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            while True:
                sender, message_code, payload = PackageProcessor.recv_package()
                if message_code == MessageCode.ParameterUpdate:
                    for model_parameters in payload:
                        updated = self._handler.add_model(sender, model_parameters)
                
                    if updated:
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))
