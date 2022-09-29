from io import StringIO
import unittest
from unittest.mock import patch

import torch

from fedlab.models.mlp import MLP
from fedlab.core.standalone import StandalonePipeline
from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.core.server.handler import ServerHandler
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.dataset import PartitionedMNIST, PathologicalMNIST

class TestStandalonePipeline(StandalonePipeline):
            def __init__(self, handler, trainer):
                super(TestStandalonePipeline, self).__init__(handler, trainer)

@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class StandalonePipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = MLP(784, 10)
        cls.num_clients = 10
        cls.comm_round = 3
        cls.sample_ratio = 0.5
        data_path = '../data/mnist'
        cls.dataset = PathologicalMNIST(root=data_path, path=data_path, num_clients=cls.num_clients, preprocess=True)
        
    def setUp(self) -> None:
        # prepare client trainer
        self.trainer = SGDSerialClientTrainer(self.model, self.num_clients, cuda=True)
        self.trainer.setup_dataset(self.dataset)
        self.trainer.setup_optim(epochs=2, batch_size=16, lr=0.1)
        # prepare server handler
        self.handler = SyncServerHandler(self.model, self.comm_round, self.sample_ratio)

    def test_init(self):
        pipeline = TestStandalonePipeline(self.handler, self.trainer)
        self.assertEqual(pipeline.trainer.num_clients, self.num_clients)
        self.assertEqual(pipeline.handler.num_clients, self.num_clients)
        self.assertIsInstance(pipeline.handler, ServerHandler)
        self.assertIsInstance(pipeline.trainer, SerialClientTrainer)

    def test_main(self):
        pipeline = TestStandalonePipeline(self.handler, self.trainer)
        pipeline.main()


    def test_evaluate(self):
        pipeline = TestStandalonePipeline(self.handler, self.trainer)
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            expected_output = "Implement your evaluation here.\n"
            pipeline.evaluate()
            self.assertEqual(mock_stdout.getvalue(), expected_output)
