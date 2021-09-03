import argparse
import os
import pickle

import torch
import torch.distributed as dist

torch.manual_seed(0)

from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict
from fedlab.core.client.scale.trainer import SerialTrainer

import sys

sys.path.append('../../../../../')
from fedlab_benchmarks.models.rnn import RNN_Shakespeare
from fedlab_benchmarks.datasets.leaf_data_process.dataloader import get_LEAF_dataloader


class RNNSTrainer(SerialTrainer):
    def __init__(self, model, client_num, aggregator, cuda=True, logger=None, args=None):
        super().__init__(model,
                         client_num,
                         aggregator,
                         cuda=cuda,
                         logger=logger)
        self.args = args

    def _get_dataloader(self, client_id):

        rank = dist.get_rank()
        client_id = (rank - 1) * self.client_num + client_id

        dataset_pkl_path = "client" + str(client_id) + ".pkl"
        with open("./pkl_dataset/train/"+dataset_pkl_path, "rb") as f:
            dataset = pickle.load(f)

        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args["batch_size"])
        return trainloader

    def _train_alone(self, model_parameters, train_loader):

        epochs, lr = self.args["epochs"], self.args["lr"]
        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._model.train()

        for _ in range(epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                output = self.model(data)

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3002")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)

    parser.add_argument("--partition", type=str, default="iid")
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str, default=None)

    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.cuda = False

    model = RNN_Shakespeare()

    aggregator = Aggregators.fedavg_aggregate

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)

    trainer = RNNSTrainer(model=model,
                          client_num=66,
                          aggregator=aggregator,
                          args={
                              "batch_size": 100,
                              "lr": 0.01,
                              "epochs": 1
                          })

    manager_ = ScaleClientPassiveManager(handler=trainer, network=network)

    manager_.run()