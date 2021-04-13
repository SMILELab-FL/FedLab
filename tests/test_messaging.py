# -*- coding: utf-8 -*-
# @Time    : 4/11/21 11:50 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : test_messaging.py
# @Software: PyCharm
import os
import unittest
from fedlab_core.message_processor import MessageCode


class MessageCodeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.message_codes = [MessageCode.ParameterRequest,
                             MessageCode.GradientUpdate,
                             MessageCode.ParameterUpdate,
                             MessageCode.EvaluateParams,
                             MessageCode.Exit]
        cls.code_names = ['ParameterRequest',
                          'GradientUpdate',
                          'ParameterUpdate',
                          'EvaluateParams',
                          'Exit']

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
