import torch
import argparse
import sys
import os

sys.path.append('../../../')
# sys.path.append('/home/zengdun/FedLab/')
from torch import nn
from fedlab_utils.logger import logger
from fedlab_core.client.topology import ClientPassiveTopology
from fedlab_core.client.trainer import ClientSGDTrainer
from fedlab_utils.models.lenet import LeNet
from fedlab_utils.models.cnn import CNN_DropOut
from fedlab_utils.models.rnn import RNN_Shakespeare
from fedlab_core.network import DistNetwork
from fedlab_utils.dataset.mnist.dataloader import get_dataloader_mnist
from fedlab_utils.dataset.femnist.dataloader import get_dataloader_femnist
from fedlab_utils.dataset.shakespeare.dataloader import get_dataloader_shakespeare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3002')
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--local_rank', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='femnist')

    args = parser.parse_args()
    args.cuda = False

    if args.dataset == 'shakespeare':
        model = RNN_Shakespeare()
        trainloader, testloader = get_dataloader_shakespeare(client_id=args.local_rank - 1)
    elif args.dataset == 'femnist':
        model = LeNet(out_dim=62)
        # model = CNN_DropOut(False)
        trainloader, testloader = get_dataloader_femnist(client_id=args.local_rank - 1)
    else:
        model = LeNet()
        trainloader, testloader = get_dataloader_mnist(args)

    if args.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    handler = ClientSGDTrainer(model, trainloader, epoch=2, optimizer=optimizer, criterion=criterion, cuda=args.cuda)
    network = DistNetwork(address=(args.server_ip, args.server_port),
                          world_size=args.world_size,
                          rank=args.local_rank)
    topology = ClientPassiveTopology(handler=handler, network=network)
    topology.run()
