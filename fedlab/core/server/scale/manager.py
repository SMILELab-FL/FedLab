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

from ...server.manager import ServerSynchronousManager
from ...coordinator import Coordinator
from ...communicator.processor import PackageProcessor
from ...communicator.package import Package
from ....utils.message_code import MessageCode

class ScaleSynchronousManager(ServerSynchronousManager):
    """ServerManager used in scale scenario."""
    def __init__(self, network, handler):
        super().__init__(network, handler)

    def activate_clients(self):
        """Add client id map"""
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client Activation Procedure")
        for rank, values in rank_dict.items():
            self._LOGGER.info("rank {}, client ids {}".format(rank, values))

            # Send parameters
            param_pack = Package(message_code=MessageCode.ParameterUpdate,
                                 content=self._handler.model_parameters)
            PackageProcessor.send_package(package=param_pack, dst=rank)

            # Send activate id list
            id_list = torch.Tensor(values).int()
            act_pack = Package(message_code=MessageCode.ParameterUpdate,
                               content=id_list,
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
