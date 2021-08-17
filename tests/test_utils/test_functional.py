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

from fedlab.utils.functional import AverageMeter, evaluate, get_best_gpu
from fedlab.utils.functional import read_config_from_json

from tests.test_core.task_setting_for_test import criterion, unittest_dataloader, model

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
        
        assert test.avg == sum/test_case and test.count == test_case and test.sum == sum
        test.reset()
        assert test.avg == 0.0 and test.count == 0.0 and test.sum == 0.0 and test.val == 0.0

    def test_read_config_json(self):
        json_file = '../data/config.json'
        json_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), json_file)
        test_config = ('127.0.0.1', '3002', 3, 0)
        self.assertEqual(test_config, read_config_from_json(json_file=json_file, user_name='server'))
        test_config = ('127.0.0.1', '3002', 3, 1)
        self.assertEqual(test_config, read_config_from_json(json_file=json_file, user_name='client_0'))
        self.assertRaises(KeyError, lambda: read_config_from_json(json_file=json_file, user_name='client_2'))