# -*- coding: utf-8 -*-
# @Time    : 4/12/21 9:05 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : test_bench.py
# @Software: PyCharm
import os
import unittest
import tests

if __name__ == '__main__':
    suite = tests.get_tests()
    unittest.TextTestRunner(verbosity=2).run(suite)
