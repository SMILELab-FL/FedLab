from io import StringIO
import unittest
from unittest.mock import patch

import torch

from .task_setting_for_test import CNN_Mnist

from fedlab.core.standalone import StandalonePipeline
from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.core.server.handler import ServerHandler
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.dataset import PartitionedMNIST

class TestStandalonePipeline(StandalonePipeline):
            def __init__(self, handler, trainer):
                super(TestStandalonePipeline, self).__init__(handler, trainer)

@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class StandalonePipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = CNN_Mnist()
        cls.num_clients = 10
        cls.comm_round = 3
        cls.sample_ratio = 0.5
        data_path = '../data/mnist'
        cls.dataset = PartitionedMNIST(root=data_path, path=data_path, num_clients=cls.num_clients, download=True, partition="iid", verbose=False)
        cls.dataset.preprocess()
        
    def setUp(self) -> None:
        
        # prepare client trainer
        self.trainer = SGDSerialClientTrainer(self.model, self.num_clients, cuda=True)
        self.trainer.setup_dataset(self.dataset)
        self.trainer.setup_optim(epochs=2, batch_size=16, lr=0.1)
        # prepare server handler
        self.handler = SyncServerHandler(self.model, self.comm_round, self.sample_ratio)
        
    # def tearDown(self):
    #     pass

    def test_init(self):
        pipeline = TestStandalonePipeline(self.handler, self.trainer)
        self.assertEqual(pipeline.trainer.num_clients, self.num_clients)
        # self.assertEqual(pipeline.handler.num_clients, self.num_clients)
        # self.assertIsInstance(pipeline.handler, ServerHandler)
        # self.assertIsInstance(pipeline.trainer, SerialClientTrainer)

    def test_main(self):
        pass

    # def test_whole_pipeline(self):
    #     try:
    #         self.init_step()
    #     except Exception as e:
    #         self.fail(f"init_step() failed ({type(e)}: {e})")


    def test_evaluate(self):
        pipeline = TestStandalonePipeline(self.handler, self.trainer)
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            expected_output = "Implement your evaluation here."
            pipeline.evaluate()
            self.assertEqual(mock_stdout, expected_output)
