import sys
import argparse

import torch
import torchvision
from torchvision import transforms

sys.path.append('../../../../../')

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from fedlab.utils.functional import AverageMeter, evaluate

from fedlab_benchmarks.models.cnn import CNN_Mnist


def write_file(acces, losses, args):
    record = open(args.name + ".txt", "w")
    record.write(str(args) + "\n")
    record.write(str(losses) + "\n")
    record.write(str(acces) + "\n")
    record.close()

class RecodeHandler(SyncParameterServerHandler):
    def __init__(self,
                 model,
                 client_num_in_total,
                 test_loader,
                 global_round=5,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=None,
                 args):
        super().__init__(model,
                         client_num_in_total,
                         global_round=global_round,
                         cuda=cuda,
                         sample_ratio=sample_ratio,
                         logger=logger)

        self.test_loader = test_loader
        self.loss_ = []
        self.acc_ = []
        self.args = args

    def _update_model(self, model_parameters_list):
        super()._update_model(model_parameters_list)

        loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                             self.test_loader)

        self.loss_.append(loss)
        self.acc_.append(acc)

        write_file(self.acc_, self.loss_)


# python server.py --world_size 11
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=100)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=0.1)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    model = CNN_Mnist()

    testset = torchvision.datasets.MNIST(
        root='../../../../datasets/data/mnist/',
        train=False,
        download=True,
        transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=int(len(testset) / 10),
                                             drop_last=False,
                                             shuffle=False)

    handler = RecodeHandler(model,
                            client_num_in_total=1,
                            global_round=args.round,
                            sample_ratio=args.sample,
                            test_loader=testloader,
                            cuda=True,
                            args=args)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler)
    manager_.run()
