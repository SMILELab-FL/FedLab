import os
import argparse
import random
from copy import deepcopy
import torchvision.transforms as transforms
from torch import nn
import torchvision
import sys
import torch

sys.path.append("../../../../")

from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing
from fedlab.utils.functional import evaluate
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.models.lenet import LeNet
from fedlab.core.client.trainer import SerialTrainer

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int)

parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs",type=int)
parser.add_argument("--partition", type=str)

# cuda config
parser.add_argument("--gpu", type=str, default="0,1,2,3")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

model = LeNet().cuda()
num_per_round = int(args.total_client*args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
criterion = torch.nn.CrossEntropyLoss()

# get raw dataset
root = "../../../../../datasets/mnist/"

# test
trainset = torchvision.datasets.MNIST(
    root=root, train=True, download=True, transform=transforms.ToTensor()
)
testset = torchvision.datasets.MNIST(
    root=root, train=False, download=True, transform=transforms.ToTensor()
)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset), drop_last=False, shuffle=False
)

# partion dataset for clients
total_client_num = args.total_client  # client总数

if args.partition == "noniid":  
    data_indices = noniid_slicing(trainset, num_clients=args.total_client, num_shards=200)
elif args.partition == "iid":
    data_indices = random_slicing(trainset, num_clients=args.total_client)
else:
    raise ValueError("invalid partition type ", args.partition)
    
local_model = deepcopy(model)
criterion = nn.CrossEntropyLoss()

# use fedlab modeles
trainer = SerialTrainer(
    model=local_model, dataset=trainset, data_slices=data_indices, aggregator=aggregator, args=args
)

model_params = SerializationTool.serialize_model(model)
to_select = [i + 1 for i in range(total_client_num)]  # client_id 从1开始

losses = []
acces = []

for round in range(args.com_round):
    selection = random.sample(to_select, num_per_round)
    print(selection)
    aggregated_parameters = trainer.train(
        model_parameters=model_params, id_list=selection, aggregate=True
    )

    SerializationTool.deserialize_model(model, aggregated_parameters)
    loss, acc = evaluate(model, criterion, test_loader)
    print("loss: {:.4f}, acc: {:.2f}".format(loss, acc))

    losses.append(loss)
    acces.append(acc)

f = open("loss.txt", "w")
f.write(str(losses))
f.close()

f = open("accuracy.txt", "w")
f.write(str(acces))
f.close()
