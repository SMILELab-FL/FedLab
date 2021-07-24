import unittest
from fedlab_utils.functional import AverageMeter
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