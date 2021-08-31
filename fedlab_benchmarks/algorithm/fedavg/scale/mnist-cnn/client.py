import torch
import argparse
import sys
import os

import torchvision
import torchvision.transforms as transforms

sys.path.append("../../../../../")

from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork

from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict

from fedlab_benchmarks.models.cnn import CNN_Mnist

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3002")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)

    parser.add_argument("--partition", type=str, default="noniid")

    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str, default=None)

    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.cuda = False

    trainset = torchvision.datasets.MNIST(
        root='../../../../datasets/data/mnist/',
        train=True,
        download=True,
        transform=transforms.ToTensor())

    if args.partition == "noniid":
        data_indices = load_dict("mnist_noniid.pkl")
    elif args.partition == "iid":
        data_indices = load_dict("mnist_iid.pkl")
    else:
        raise ValueError("invalid partition type ", args.partition)

    # Process rank x represent client id from (x-1)*10 - (x-1)*10 +10
    # e.g. rank 5 <--> client 40-50
    client_id_list = [
        i for i in range((args.rank - 1) * 10, (args.rank - 1) * 10 + 10)
    ]

    # get corresponding data partition indices
    sub_data_indices = {
        idx: data_indices[cid]
        for idx, cid in enumerate(client_id_list)
    }

    model = CNN_Mnist()
    
    aggregator = Aggregators.fedavg_aggregate

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)

    trainer = SubsetSerialTrainer(model=model,
                            dataset=trainset,
                            data_slices=sub_data_indices,
                            aggregator=aggregator,
                            args={
                                "batch_size": 100,
                                "lr": 0.02,
                                "epochs": 5
                            })

    manager_ = ScaleClientPassiveManager(handler=trainer, network=network)

    manager_.run()