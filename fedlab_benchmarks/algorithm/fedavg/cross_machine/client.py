from logging import log
import torch
import argparse
import sys
import os

sys.path.append('../../../../')

from torch import nn
from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import logger

from setting import get_model, get_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--server_ip', type=str)
    parser.add_argument('--server_port', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--rank', type=int)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--ethernet", type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = get_model(args)
    trainloader, testloader = get_dataset(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    network = DistNetwork(address=(args.server_ip, args.server_port),
                          world_size=args.world_size,
                          rank=args.rank,
                          ethernet=args.ethernet)
    LOGGER = logger(log_name="client " + str(args.rank))

    handler = ClientSGDTrainer(model,
                               trainloader,
                               epoch=args.epoch,
                               optimizer=optimizer,
                               criterion=criterion,
                               cuda=args.cuda,
                               logger=LOGGER)

    manager_ = ClientPassiveManager(handler=handler,
                                    network=network,
                                    logger=LOGGER)
    manager_.run()
