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


import os
import unittest
from fedlab.utils.message_code import MessageCode


class MessageCodeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.message_codes = [MessageCode.ParameterRequest,
                             MessageCode.GradientUpdate,
                             MessageCode.ParameterUpdate,
                             MessageCode.EvaluateParams,
                             MessageCode.Exit,
                             MessageCode.SetUp]
        cls.code_names = ['ParameterRequest',
                          'GradientUpdate',
                          'ParameterUpdate',
                          'EvaluateParams',
                          'Exit',
                          'SetUp']

    def test_message_code_eq(self):
        num = len(self.message_codes)
        for i in range(num):
            self.assertEqual(self.message_codes[i], MessageCode[self.code_names[i]])

    def test_message_code_name(self):
        num = len(self.message_codes)
        for i in range(num):
            self.assertEqual(self.message_codes[i].name, self.code_names[i])

    def test_message_code_value(self):
        num = len(self.message_codes)
        for i in range(num):
            self.assertEqual(i, self.message_codes[i].value)
