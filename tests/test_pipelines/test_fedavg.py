import os
import unittest
import torch

class FedAvgTestCase(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "torch.cuda is required")
    def test_fedavg_pipeline(self):
        os.chdir(os.path.join(os.getcwd(),"tests","test_pipelines"))
        os.system("bash test_fedavg.sh")