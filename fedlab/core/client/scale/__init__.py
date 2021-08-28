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

from ...client.manager import ClientPassiveManager

from ...communicator.package import Package
from ...communicator.processor import PackageProcessor

from fedlab.utils.message_code import MessageCode

class ScaleClientManager(ClientPassiveManager):
    def __init__(self, handler, network):
        super().__init__(network=network, handler=handler)

    def setup(self):
        super().setup()
        content = torch.Tensor([self._handler.client_num]).int()
        setup_pack = Package(content=content, data_type=1)
        PackageProcessor.send_package(setup_pack, dst=0)

    def on_receive(self, sender_rank, message_code, payload):
        if message_code == MessageCode.ParameterUpdate:
            model_parameters = payload[0]
            print("model size:", model_parameters.shape)
            _, message_code, payload = PackageProcessor.recv_package(src=0)
            id_list = payload[0].tolist()
            self.model_parameters_list = self._handler.train(
                model_parameters=model_parameters,
                id_list=id_list,
                aggregate=False)

    def synchronize(self):
        pack = Package(message_code=MessageCode.ParameterUpdate,
                       content=self.model_parameters_list)
        PackageProcessor.send_package(package=pack, dst=0)