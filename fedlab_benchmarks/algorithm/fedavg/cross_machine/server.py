import os
import sys
import argparse

sys.path.append('../../../../')
from fedlab.utils.logger import Logger
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork
from setting import get_model
import torch
from setting import get_dataset
import torchvision
from torchvision import transforms
from fedlab.utils.functional import evaluate

class TestCustomizationServer(SyncParameterServerHandler):
    def __init__(self, model: torch.nn.Module, client_num_in_total: int, global_round, cuda, sample_ratio, logger):
        super().__init__(model, client_num_in_total, global_round=global_round, cuda=cuda, sample_ratio=sample_ratio, logger=logger)
        
        root = '../../../../../datasets/mnist/'
        testset = torchvision.datasets.MNIST(root=root,
                                             train=False,
                                             download=True,

                                             transform=transforms.ToTensor())
        self.test_loader = torch.utils.data.DataLoader(testset,
                                                 batch_size=int(len(testset)/10),
                                                 drop_last=False,
                                                 num_workers=2,
                                                 shuffle=False)

        self.test_loss = torch.nn.CrossEntropyLoss()                              

    def _update_model(self, serialized_params_list):
        self._LOGGER.info("updating global model")
        super()._update_model(serialized_params_list)

        loss_, acc = evaluate(self._model, self.test_loss, self.test_loader, cuda=True)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--server_ip', type=str)
    parser.add_argument('--server_port', type=str)
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ethernet', type=str)
    parser.add_argument('--sample', type=float)

    args = parser.parse_args()

    model = get_model(args)
    LOGGER = Logger(log_name="server")

    #ps = SyncParameterServerHandler(model, client_num_in_total=args.world_size-1, global_round=args.round, logger=LOGGER, sample_ratio=args.sample)
    ps = TestCustomizationServer(model, client_num_in_total=args.world_size-1, global_round=args.round, logger=LOGGER, sample_ratio=args.sample, cuda=True)
    
    network = DistNetwork(address=(args.server_ip, args.server_port), world_size=args.world_size, rank=0, ethernet=args.ethernet)
    manager_ = ServerSynchronousManager(handler=ps, network=network, logger=LOGGER)
    manager_.run()
 