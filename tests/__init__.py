# -*- coding: utf-8 -*-
# @Time    : 3/19/21 6:32 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : __init__.py
# @Software: PyCharm
import unittest


def get_tests():
    from .test_utils.test_serialization import SerializationTestCase
    from .test_utils.test_message_code import MessageCodeTestCase
    from .test_utils.test_aggregator import AggregatorTestCase
    from .test_utils.test_functional import FunctionalTestCase
    from .test_utils.test_logger import LoggerTestCase
    

    
    from .test_core.test_processor import PackageTestCase
    from .test_core.test_compressor import CompressorTestCase
    from .test_core.test_handler import HandlerTestCase
    from .test_core.test_network import NetworkTestCase
    from .test_core.test_topology import TopologyTestCase

    serialization_suite = unittest.TestLoader().loadTestsFromTestCase(SerializationTestCase)
    message_code_suite = unittest.TestLoader().loadTestsFromTestCase(MessageCodeTestCase)
    package_suite = unittest.TestLoader().loadTestsFromTestCase(PackageTestCase)
    functional_suite = unittest.TestLoader().loadTestsFromTestCase(FunctionalTestCase)
    logger_suite = unittest.TestLoader().loadTestsFromTestCase(LoggerTestCase)
    aggregator_suite = unittest.TestLoader().loadTestsFromTestCase(AggregatorTestCase)
    compressor_suite = unittest.TestLoader().loadTestsFromTestCase(CompressorTestCase)
    handler_suite = unittest.TestLoader().loadTestsFromTestCase(HandlerTestCase)
    network_suite = unittest.TestLoader().loadTestsFromTestCase(NetworkTestCase)
    topology_suite = unittest.TestLoader().loadTestsFromTestCase(TopologyTestCase)

    return unittest.TestSuite([serialization_suite,
                               message_code_suite,
                               package_suite,
                               aggregator_suite,
                               compressor_suite,
                               handler_suite,
                               functional_suite,
                               logger_suite,
                               network_suite,
                               topology_suite])
