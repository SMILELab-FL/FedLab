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

from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.core.client.scale.manager import ScaleClientPassiveManager
from fedlab.core.network import DistNetwork
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import load_dict

from fedlab_benchmarks.models.cnn import CNN_Femnist

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

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='../../../../datasets/data/cifar10/',
        train=True,
        download=True,
        transform=transform_train)

    if args.partition == "noniid":
        data_indices = load_dict("cifar10_noniid.pkl")
    elif args.partition == "iid":
        data_indices = load_dict("cifar10_iid.pkl")
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

    model = CNN_Femnist()

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
                                "lr": 0.001,
                                "epochs": 5
                            })

    manager_ = ScaleClientPassiveManager(handler=trainer, network=network)

    manager_.run()