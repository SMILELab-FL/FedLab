import torch
import argparse
import sys

sys.path.append('../../../')

from torch import nn
from fedlab_core.client.manager import ClientPassiveManager
from fedlab_core.client.trainer import ClientSGDTrainer
from fedlab_utils.models.lenet import LeNet
from fedlab_utils.models.rnn import RNN_Shakespeare
from fedlab_utils.models.rnn import RNN_Sent140
from fedlab_core.network import DistNetwork
from fedlab_utils.dataset.leaf.dataloader import get_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3002')
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--local_rank', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='sent140')

    args = parser.parse_args()
    args.cuda = True

    if args.dataset == 'shakespeare':
        model = RNN_Shakespeare()
        trainloader, testloader = get_dataloader(dataset=args.dataset, client_id=args.local_rank - 1)
    elif args.dataset == 'femnist':
        model = LeNet(out_dim=62)
        # model = CNN_DropOut(False)
        trainloader, testloader = get_dataloader(dataset=args.dataset, client_id=args.local_rank - 1)
    elif args.dataset == 'sent140':
        model = RNN_Sent140()
        trainloader, testloader = get_dataloader(dataset=args.dataset, client_id=args.local_rank - 1)
    else:
        pass

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
    Manager = ClientPassiveManager(handler=handler, network=network)
    Manager.run()
