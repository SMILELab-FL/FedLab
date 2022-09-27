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

import unittest
import argparse
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import random

from fedlab.utils.functional import get_best_gpu
from fedlab.core.client import ORDINARY_TRAINER, SERIAL_TRAINER
from fedlab.core.client.trainer import ClientTrainer, SerialClientTrainer
from fedlab.utils.dataset.functional import noniid_slicing, random_slicing
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool

from ..task_setting_for_test import CNN_Mnist


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ClientTrainerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        class TestClientTrainer(ClientTrainer):
            def __init__(self, model, cuda=True, device=None):
                super().__init__(model, cuda, device)

        self.model = CNN_Mnist()
        self.cuda = True
        self.device = 'cuda'
        self.trainer = TestClientTrainer(model=self.model, cuda=self.cuda, device=self.device)

    def tearDown(self) -> None:
        return super().tearDown()

    
    def test_init(self):
        self.assertEqual(self.trainer.cuda, self.cuda)
        self.assertEqual(self.trainer.device, self.device)
        self.assertEqual(self.trainer.type, ORDINARY_TRAINER)

    
    def test_setup_dataset(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.setup_dataset()


    def test_setup_optim(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.setup_optim()


    def test_uplink_package(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.uplink_package


    def test_local_process(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.local_process([None])

    def test_train(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.train()

    def test_validate(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.validate()

    def test_evaluate(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.evaluate()

    

@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SerialClientTrainerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        class TestSerialClientTrainer(SerialClientTrainer):
            def __init__(self, model, num_clients, cuda, device=None, personal=False):
                super().__init__(model, num_clients, cuda, device, personal)

        self.model = CNN_Mnist()
        self.cuda = True
        self.device = 'cuda'
        self.num_clients = 10
        self.personal = True
        self.trainer = TestSerialClientTrainer(model=self.model, num_clients=self.num_clients, cuda=self.cuda, device=self.device, personal=self.personal)

    def tearDown(self) -> None:
        return super().tearDown()

    
    def test_init(self):
        self.assertEqual(self.trainer.cuda, self.cuda)
        self.assertEqual(self.trainer.device, self.device)
        self.assertEqual(self.trainer.type, SERIAL_TRAINER)
        self.assertEqual(self.trainer.num_clients, self.num_clients)
        # self.assertEqual(self.trainer.personal, self.personal)
        if self.personal is True:
            self.assertEqual(len(self.trainer.parameters), self.num_clients)
        else:
            self.assertEqual(self.trainer.parameters, None)

    
    def test_setup_dataset(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.setup_dataset()


    def test_setup_optim(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.setup_optim()


    def test_uplink_package(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.uplink_package


    def test_local_process(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.local_process(id_list=[1,2], payload=[None, None])

    def test_train(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.train()

    def test_validate(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.validate()

    def test_evaluate(self):
        with self.assertRaises(NotImplementedError):
            self.trainer.evaluate()