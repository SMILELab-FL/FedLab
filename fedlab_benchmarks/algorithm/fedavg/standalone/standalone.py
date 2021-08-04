import os
import argparse
import random
from copy import deepcopy
import torchvision.transforms as transforms
from torch import nn
import torchvision
import sys
import torch

sys.path.append('../../../../')

from fedlab_utils.dataset.slicing import noniid_slicing
from fedlab_utils.functional import evaluate
from fedlab_utils.serialization import SerializationTool
from fedlab_utils.aggregator import Aggregators
from fedlab_utils.models.lenet import LeNet
from fedlab_core.client.serial_trainer import SerialTrainer

# configuration
# server config
parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--total_client', type=int, default=100)
parser.add_argument('--num_per_round', type=int, default=10)
parser.add_argument('--com_round', type=int)

# local config
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)

# cuda config
parser.add_argument('--gpu', type=str, default=0)
parser.add_argument('--wandb', type=bool, default=False)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

model = LeNet().cuda()
aggregator = Aggregators.fedavg_aggregate
criterion = torch.nn.CrossEntropyLoss()

# get raw dataset
root = '../../../../datasets/mnist/'
# test
trainset = torchvision.datasets.MNIST(root=root,
                                      train=True,
                                      download=True,
                                      transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root=root,
                                     train=False,
                                     download=True,
                                     transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=len(testset),
                                          drop_last=False,
                                          shuffle=False)

# partion dataset for clients
total_client_num = args.total_client  # client总数
data_indices = noniid_slicing(trainset,
                              num_clients=args.total_client,
                              num_shards=200)

local_model = deepcopy(model)
criterion = nn.CrossEntropyLoss()

# use fedlab modeles
serial_handler = SerialTrainer(model=local_model,
                               dataset=trainset,
                               data_slices=data_indices,
                               aggregator=aggregator)
model_params = SerializationTool.serialize_model(model)
to_select = [i + 1 for i in range(total_client_num)]  # client_id 从1开始

for round in range(args.com_round):
    selection = random.sample(to_select, args.num_per_round)
    print(selection)
    model_params = serial_handler.train(model_parameters=model_params,
                                        epochs=args.epochs,
                                        lr=args.lr,
                                        batch_size=args.batch_size,
                                        id_list=selection,
                                        cuda=True,
                                        multi_threading=False)
    SerializationTool.deserialize_model(model, model_params)
    loss, acc = evaluate(model, criterion, test_loader, cuda=True)
