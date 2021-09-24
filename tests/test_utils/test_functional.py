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
import random
import os
import numpy as np

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from fedlab.utils.functional import AverageMeter, get_best_gpu, evaluate, save_dict, load_dict
from fedlab.utils.functional import read_config_from_json, partition_report


class FunctionalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_AvgMeter(self):
        test = AverageMeter()
        test_case = 50
        sum = 0.0
        for _ in range(test_case):
            sample = random.random()
            test.update(val=sample)
            sum += sample
            assert test.val == sample

        assert (test.avg == sum / test_case and test.count == test_case
                and test.sum == sum)
        test.reset()
        assert (test.avg == 0.0 and test.count == 0.0 and test.sum == 0.0
                and test.val == 0.0)

    def test_read_config_json(self):
        json_file = "../data/config.json"
        json_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 json_file)
        test_config = ("127.0.0.1", "3002", 3, 0)
        self.assertEqual(
            test_config,
            read_config_from_json(json_file=json_file, user_name="server"))
        test_config = ("127.0.0.1", "3002", 3, 1)
        self.assertEqual(
            test_config,
            read_config_from_json(json_file=json_file, user_name="client_0"),
        )
        self.assertRaises(
            KeyError,
            lambda: read_config_from_json(json_file=json_file,
                                          user_name="client_2"),
        )

    def test_dict(self):
        test_dict = {1: [1, 2, 3, 4, 5], 2: [6, 7, 8, 9, 10]}

        save_dict(test_dict, "./test.pkl")

        check_dict = load_dict("./test.pkl")

        assert test_dict == check_dict
        os.remove("./test.pkl")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_evaluate(self):
        class mlp(nn.Module):
            def __init__(self):
                super(mlp, self).__init__()
                self.fc1 = nn.Linear(784, 200)
                self.fc2 = nn.Linear(200, 200)
                self.fc3 = nn.Linear(200, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = x.view(x.shape[0], -1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = mlp()
        root = "./tests/data/mnist/"
        testset = torchvision.datasets.MNIST(root=root,
                                             train=False,
                                             download=False,
                                             transform=transforms.ToTensor())
        criterion = nn.CrossEntropyLoss()
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=len(testset),
                                                  drop_last=False,
                                                  shuffle=False)
        evaluate(model, criterion, test_loader)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_get_gpu(self):
        gpu = get_best_gpu()

    def test_partition_report(self):
        np.random.seed(0)
        data_indices = {0: np.array([0, 1, 2]),
                        1: np.array([3, 4, 5]),
                        2: np.array([6, 7, 8])}
        labels = np.array([0] * 3 + [1] * 3 + [2] * 3)
        labels = labels[np.random.permutation(9)]
        file = os.path.join(os.path.dirname(__file__), "tmp.csv")
        partition_report(labels, data_indices, class_num=3, verbose=True, file=None)
        partition_report(labels, data_indices, class_num=3, verbose=False, file=file)
        self.assertTrue(os.path.exists(file))
        if os.path.exists(file):
            os.remove(file)
