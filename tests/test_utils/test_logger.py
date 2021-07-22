import unittest
import os
from fedlab_utils.logger import logger



class LoggerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_logger(self):
        
        LOGGER = logger("./test.txt", log_name="test")
        os.remove("./test.txt")
        
