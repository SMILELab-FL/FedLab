import argparse
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

torch.manual_seed(0)

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import evaluate
from torch.utils.data import ConcatDataset

import sys

sys.path.append('../../../../../')
from fedlab_benchmarks.models.rnn import RNN_Shakespeare


def write_file(acces, losses, name="shakespeare_noniid"):
    record = open(name + ".txt", "w")

    record.write(str(losses) + "\n\n")
    record.write(str(acces) + "\n\n")
    record.close()


class RecodeHandler(SyncParameterServerHandler):
    def __init__(self,
                 model,
                 client_num_in_total,
                 test_loader,
                 global_round=5,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=None):
        super().__init__(model,
                         client_num_in_total,
                         global_round=global_round,
                         cuda=cuda,
                         sample_ratio=sample_ratio,
                         logger=logger)

        self.test_loader = test_loader
        self.loss_ = []
        self.acc_ = []

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

    parser.add_argument('--round', type=int, default=1000)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=0.1)

    args = parser.parse_args()

    model = RNN_Shakespeare()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    print("creating global test set")
    dataset_list = []
    for i in range(660):
        file_name = "client" + str(i) + ".pkl"
        with open("./pkl_dataset/test/" + file_name, 'rb') as f:
            test = pickle.load(f)
        dataset_list.append(test)

    testset = ConcatDataset(dataset_list)
    testloader = torch.utils.data.DataLoader(testset, batch_size=500)
    print("done")

    handler = RecodeHandler(model,
                            client_num_in_total=1,
                            global_round=args.round,
                            sample_ratio=args.sample,
                            test_loader=testloader,
                            cuda=True)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler)
    manager_.run()
