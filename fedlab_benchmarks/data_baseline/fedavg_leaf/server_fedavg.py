import os
import sys

sys.path.append('../../../')
# sys.path.append('/home/zengdun/FedLab/')

from fedlab_utils.logger import logger
from fedlab_utils.models.lenet import LeNet
from fedlab_utils.models.cnn import CNN_DropOut
from fedlab_utils.models.rnn import RNN_Shakespeare
from fedlab_core.server.handler import SyncParameterServerHandler
from fedlab_core.server.topology import ServerSynchronousTopology
import argparse
from fedlab_core.network import DistNetwork


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='3002')
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='femnist')
    args = parser.parse_args()

    args.cuda = False
    if args.dataset == 'shakespeare':
        model = RNN_Shakespeare()
    elif args.dataset == 'femnist':
        model = LeNet(out_dim=62)
        # model = CNN_DropOut(False)
    else:
        model = LeNet()

    if args.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    ps = SyncParameterServerHandler(model, client_num_in_total=args.world_size-1)

    network = DistNetwork(address=(args.server_ip, args.server_port), world_size=args.world_size, rank=0)
    topology = ServerSynchronousTopology(handler=ps, network=network)
        
    topology.run()
