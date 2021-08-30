import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
import sys
import torch

torch.manual_seed(0)

sys.path.append("../../../../")

from fedlab.core.client.scale.trainer import SubsetSerialTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing
from fedlab.utils.functional import get_best_gpu

from fedlab_benchmarks.models.cnn import CNN_Mnist

# python standalone.py --com_round 10 --sample_ratio 0.1 --batch_size 10 --epochs 5 --partition iid --lr 0.02

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int, default=5000)

parser.add_argument("--sample_ratio", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--partition", type=str, default='iid')

args = parser.parse_args()

# get raw dataset
root = "../../../../../datasets/mnist/"
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

# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

gpu = get_best_gpu()
model = CNN_Mnist().cuda(gpu)

# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client  # client总数

if args.partition == "noniid":
    data_indices = noniid_slicing(trainset,
                                  num_clients=args.total_client,
                                  num_shards=200)
elif args.partition == "iid":
    data_indices = random_slicing(trainset, num_clients=args.total_client)
else:
    raise ValueError("invalid partition type ", args.partition)

# fedlab setup
local_model = deepcopy(model)

trainer = SubsetSerialTrainer(model=local_model,
                              dataset=trainset,
                              data_slices=data_indices,
                              aggregator=aggregator,
                              args=args)

loss_ = []
acc_ = []

# train procedure
to_select = [i for i in range(total_client_num)]
for round in range(args.com_round):
    model_parameters = SerializationTool.serialize_model(model)
    selection = random.sample(to_select, num_per_round)
    print(selection)
    aggregated_parameters = trainer.train(model_parameters=model_parameters,
                                          id_list=selection,
                                          aggregate=True)

    SerializationTool.deserialize_model(model, aggregated_parameters)

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    print("loss: {:.4f}, acc: {:.2f}".format(loss, acc))

    loss_.append(loss)
    acc_.append(acc)