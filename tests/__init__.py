# -*- coding: utf-8 -*-
# @Time    : 3/19/21 6:32 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : __init__.py
# @Software: PyCharm
import unittest


def get_tests():
    from .test_serialization import SerializationTestCase

    serialization_suite = unittest.TestLoader().loadTestsFromTestCase(SerializationTestCase)

    return unittest.TestSuite([serialization_suite])
