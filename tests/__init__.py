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


def get_tests():

    from .test_utils.test_serialization import SerializationTestCase
    from .test_utils.test_message_code import MessageCodeTestCase
    from .test_utils.test_aggregator import AggregatorTestCase
    from .test_utils.test_functional import FunctionalTestCase
    from .test_utils.test_logger import LoggerTestCase

    from .test_core.test_communicator.test_processor import ProcessorTestCase
    from .test_core.test_communicator.test_compressor import CompressorTestCase
    from .test_core.test_server.test_parameter_server_handler import HandlerTestCase
    from .test_core.test_network import NetworkTestCase
    from .test_core.test_network_manager import ManagerTestCase
    from .test_core.test_communicator.test_package import PackageTestCase

    from .test_pipelines.test_fedavg import FedAvgTestCase
    
    serialization_suite = unittest.TestLoader().loadTestsFromTestCase(SerializationTestCase)
    message_code_suite = unittest.TestLoader().loadTestsFromTestCase(MessageCodeTestCase)
    processor_suite = unittest.TestLoader().loadTestsFromTestCase(ProcessorTestCase)
    functional_suite = unittest.TestLoader().loadTestsFromTestCase(FunctionalTestCase)
    logger_suite = unittest.TestLoader().loadTestsFromTestCase(LoggerTestCase)
    aggregator_suite = unittest.TestLoader().loadTestsFromTestCase(AggregatorTestCase)
    compressor_suite = unittest.TestLoader().loadTestsFromTestCase(CompressorTestCase)
    handler_suite = unittest.TestLoader().loadTestsFromTestCase(HandlerTestCase)
    network_suite = unittest.TestLoader().loadTestsFromTestCase(NetworkTestCase)
    manager_suite = unittest.TestLoader().loadTestsFromTestCase(ManagerTestCase)
    package_suite = unittest.TestLoader().loadTestsFromTestCase(PackageTestCase)

    fedavg_suite = unittest.TestLoader().loadTestsFromTestCase(FedAvgTestCase)


    return unittest.TestSuite([serialization_suite,
                               message_code_suite,
                               processor_suite,
                               aggregator_suite,
                               compressor_suite,
                               handler_suite,
                               functional_suite,
                               logger_suite,
                               network_suite,
                               manager_suite,
                               package_suite,
                               fedavg_suite])
