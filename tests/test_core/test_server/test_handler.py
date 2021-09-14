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

import random
import unittest

import torch

from ..task_setting_for_test import CNN_Mnist
from fedlab.utils.serialization import SerializationTool
from fedlab.core.server.handler import AsyncParameterServerHandler, SyncParameterServerHandler


class HandlerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = CNN_Mnist()
        cls.sample_ratio = 0.1
        cls.total_num = 100

    def setUp(self) -> None:

        self.AsyncHandler = AsyncParameterServerHandler(model=self.model)
        self.AsyncHandler.client_num_in_total = self.total_num
        self.SyncHandler = SyncParameterServerHandler(model=self.model, sample_ratio=self.sample_ratio)
        self.SyncHandler.client_num_in_total = self.total_num
        
    def tearDown(self) -> None:
        return super().tearDown()

    def test_update_model(self):
        coming_model = CNN_Mnist()
        coming_parameters = SerializationTool.serialize_model(coming_model)

        self.AsyncHandler._update_model(coming_parameters,
                                        random.randint(1, 10))

        parameter_list = []
        for id in range(self.SyncHandler.client_num_per_round):
            tensors = torch.Tensor(size=coming_parameters.shape)
            parameter_list.append(tensors)
            flag = self.SyncHandler.add_model(id, tensors)
        assert flag

        self.SyncHandler._update_model(parameter_list)

    def test_sample(self):
        samples = self.SyncHandler.sample_clients()
        assert len(samples) == max(1, int(self.sample_ratio * self.total_num))
