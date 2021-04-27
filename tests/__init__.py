# -*- coding: utf-8 -*-
# @Time    : 3/19/21 6:32 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : __init__.py
# @Software: PyCharm
import unittest


def get_tests():
    from .test_serialization import SerializationTestCase
    from .test_message_code import MessageCodeTestCase

    serialization_suite = unittest.TestLoader().loadTestsFromTestCase(SerializationTestCase)
    message_code_suite = unittest.TestLoader().loadTestsFromTestCase(MessageCodeTestCase)

    return unittest.TestSuite([serialization_suite,
                               message_code_suite])
