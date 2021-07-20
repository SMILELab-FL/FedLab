import unittest
import random
import torch
from fedlab_utils.aggregator import Aggregators


class AggregatorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fedavg_aggregator = Aggregators.fedavg_aggregate
        cls.fedasgd_aggregator = Aggregators.fedasgd_aggregate

        cls.shape = (10000,)

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_fedavg_aggregate(self):
        params = []
        for _ in range(random.randint(1, 10)):
            params.append(torch.Tensor(size=self.shape))

        merged_params = self.fedavg_aggregator(params)

        assert self.shape == merged_params.shape

    def test_fedasgd_aggregate(self):
        
        server_params = torch.Tensor(size=self.shape)
        comming_params = torch.Tensor(size=self.shape)

        merged_params = self.fedasgd_aggregator(server_params, comming_params, 0.5)

        assert self.shape == merged_params.shape

