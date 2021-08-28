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
from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool

from ..task_setting_for_test import mlp


class TrainerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.total_client = 10
        self.num_per_round = 5
        self.aggregator = Aggregators.fedavg_aggregate

        self.root = "./tests/data/mnist/"

    def tearDown(self) -> None:
        return super().tearDown()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_serial_train_iid(self):

        trainset = torchvision.datasets.MNIST(root=self.root,
                                              train=True,
                                              download=True,
                                              transform=transforms.ToTensor())
        data_indices = random_slicing(trainset, num_clients=self.total_client)
        gpu = get_best_gpu()
        model = mlp().cuda(gpu)
        trainer = SubsetSerialTrainer(
            model=model,
            dataset=trainset,
            data_slices=data_indices,
            aggregator=self.aggregator,
            args={"batch_size": 100, "epochs": 1, "lr": 0.1},
        )

        to_select = [i for i in range(self.total_client)]
        model_parameters = SerializationTool.serialize_model(model)
        selection = random.sample(to_select, self.num_per_round)
        aggregated_parameters = trainer.train(
            model_parameters=model_parameters,
            id_list=selection,
            aggregate=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_serial_train_noniid(self):

        root = "./tests/data/mnist/"
        trainset = torchvision.datasets.MNIST(root=root,
                                              train=True,
                                              download=True,
                                              transform=transforms.ToTensor())
        data_indices = data_indices = noniid_slicing(
            trainset, num_clients=self.total_client, num_shards=200)
        gpu = get_best_gpu()
        model = mlp().cuda(gpu)
        trainer = SubsetSerialTrainer(
            model=model,
            dataset=trainset,
            data_slices=data_indices,
            aggregator=self.aggregator,
            args={"batch_size": 100, "epochs": 1, "lr": 0.1},
        )
        to_select = [i for i in range(self.total_client)]
        model_parameters = SerializationTool.serialize_model(model)
        selection = random.sample(to_select, self.num_per_round)
        aggregated_parameters = trainer.train(
            model_parameters=model_parameters,
            id_list=selection,
            aggregate=True)
