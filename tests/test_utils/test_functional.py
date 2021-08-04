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
from fedlab.utils.functional import AverageMeter
import random

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
