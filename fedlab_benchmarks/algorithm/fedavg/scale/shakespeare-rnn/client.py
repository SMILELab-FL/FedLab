import torch
import argparse
import sys
import os

from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)
sys.path.append("../../../../../")

from fedlab.core.client.scale import ScaleClientManager
from fedlab.core.network import DistNetwork
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict
from fedlab.core.client.scale.trainer import SerialTrainer

from fedlab_benchmarks.models.rnn import RNN_Shakespeare
from fedlab_benchmarks.datasets.leaf_data_process.dataloader import get_LEAF_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3003")
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

    trainer = SerialTrainer(model=model,
                           aggregator=aggregator,
                           args={
                               "batch_size": 100,
                               "lr": 0.001,
                               "epochs": 5
                           })

    manager_ = ScaleClientManager(handler=trainer, network=network)

    manager_.run()