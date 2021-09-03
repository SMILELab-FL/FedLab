import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

torch.manual_seed(0)

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import AverageMeter

import sys

sys.path.append('../../../../../')

from fedlab_benchmarks.models.cnn import AlexNet_CIFAR10
from config import cifar10_noniid_baseline_config, cifar10_iid_baseline_config


def evaluate(model, criterion, test_loader):
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.sum, acc_.avg


def write_file(acces, losses, config):
    record = open(
        "{}_{}_{}.txt".format(config['partition'], config['network'],
                              config['dataset']), "w")

    record.write(str(config) + "\n")
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
                 config=None):
        super().__init__(model,
                         client_num_in_total,
                         global_round=global_round,
                         cuda=cuda,
                         sample_ratio=sample_ratio,
                         logger=logger)

        self.test_loader = test_loader
        self.loss_ = []
        self.acc_ = []
        self.config = config

    def _update_model(self, model_parameters_list):
        super()._update_model(model_parameters_list)

        loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                             self.test_loader)

        self.loss_.append(loss)
        self.acc_.append(acc)

        write_file(self.acc_, self.loss_, self.config)


# python server.py --world_size 11
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3003")
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--ethernet', type=str, default=None)

    parser.add_argument('--setting', type=str)
    args = parser.parse_args()

    model = AlexNet_CIFAR10()

    if args.setting == "iid":
        config = cifar10_iid_baseline_config
    else:
        config = cifar10_noniid_baseline_config

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='../../../../datasets/data/cifar10/',
        train=False,
        download=True,
        transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=int(len(testset) / 10),
                                             drop_last=False,
                                             shuffle=False)

    handler = RecodeHandler(model,
                            client_num_in_total=1,
                            global_round=config["round"],
                            sample_ratio=config["sample_ratio"],
                            test_loader=testloader,
                            cuda=True,
                            config=config)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler)
    manager_.run()