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
from fedlab.core.server.handler import ServerHandler


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ServerHandlerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = CNN_Mnist()

    def setUp(self) -> None:
        class TestServerHandler(ServerHandler):
            def __init__(self, model, cuda=True, device=None):
                super().__init__(model, cuda, device)

        self.cuda = True
        self.device = 'cuda'
        self.handler = TestServerHandler(model=self.model, cuda=self.cuda, device=self.device)
        
    def tearDown(self) -> None:
        return super().tearDown()

    def test_init(self):
        self.assertEqual(self.handler.cuda, self.cuda)
        self.assertEqual(self.handler.device, self.device)

    def test_downlink_package(self):
        with self.assertRaises(NotImplementedError):
            self.handler.downlink_package

    def test_if_stop(self):
        self.assertEqual(self.handler.if_stop, False)
    
    def test_global_update(self):
        with self.assertRaises(NotImplementedError):
            self.handler.global_update(None)

    def test_setup_optim(self):
        with self.assertRaises(NotImplementedError):
            self.handler.setup_optim()
    
    def test_load(self):
        with self.assertRaises(NotImplementedError):
            self.handler.load(None)

    def test_evaluate(self):
        with self.assertRaises(NotImplementedError):
            self.handler.evaluate()
            

    
