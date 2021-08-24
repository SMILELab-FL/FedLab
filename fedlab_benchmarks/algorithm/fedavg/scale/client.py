import torch
import argparse
import sys
import os

import torchvision
import torchvision.transforms as transforms

sys.path.append("../../../../")

from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.network_manager import NetworkManager
from fedlab.core.client.trainer import SerialTrainer
from fedlab.core.network import DistNetwork

from fedlab.core.communicator.package import Package
from fedlab.core.communicator.processor import PackageProcessor

from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing
from fedlab.utils.message_code import MessageCode

from setting import get_model


class ScaleClientManager(ClientPassiveManager):
    def __init__(self, handler, network):
        super().__init__(network=network, handler=handler)

    def setup(self):
        super().setup()
        content = torch.Tensor([self._handler.client_num]).int()
        setup_pack = Package(content=content, data_type=1)
        PackageProcessor.send_package(setup_pack, dst=0)

    def on_receive(self, sender_rank, message_code, payload):
        if message_code == MessageCode.ParameterUpdate:
            model_parameters = payload[0]
            _, message_code, payload = PackageProcessor.recv_package(src=0)
            id_list = payload[0].tolist()
            model_parameters_list = self._handler.train(
                model_parameters=model_parameters,
                id_list=id_list,
                aggregate=False)

            pack = Package(message_code=MessageCode.ParameterUpdate,
                           content=model_parameters_list)

            PackageProcessor.send_package(package=pack, dst=0)

    def synchronize(self):
        pass
    """
    def run(self):
        self.setup()
        sender_rank, message_code, payload = PackageProcessor.recv_package(
            src=0)
        self.on_receive(sender_rank, message_code, payload)
    """

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="3002")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)
    #parser.add_argument("--num", type=int)

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--partition", type=str, default="iid")
    parser.add_argument("--total_client", type=int, default=10)

    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str, default=None)

    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.cuda = False

    root = "../../../../../datasets/mnist/"
    trainset = torchvision.datasets.MNIST(root=root,
                                          train=True,
                                          download=True,
                                          transform=transforms.ToTensor())

    if args.partition == "noniid":
        data_indices = noniid_slicing(trainset,
                                      num_clients=args.total_client,
                                      num_shards=200)
    elif args.partition == "iid":
        data_indices = random_slicing(trainset, num_clients=args.total_client)
    else:
        raise ValueError("invalid partition type ", args.partition)

    model = get_model(args)
    aggregator = Aggregators.fedavg_aggregate

    total_client_num = args.total_client  # client总数
    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)

    LOGGER = Logger(log_name="client " + str(args.rank))

    trainer = SerialTrainer(model=model,
                            dataset=trainset,
                            data_slices=data_indices,
                            aggregator=aggregator,
                            args={
                                "batch_size": 100,
                                "lr": 0.1,
                                "epochs": 5
                            })

    manager_ = ScaleClientManager(handler=trainer, network=network)

    manager_.run()