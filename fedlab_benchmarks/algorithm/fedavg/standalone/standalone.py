from json import load
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
from fedlab.utils.functional import get_best_gpu, load_dict

from fedlab_benchmarks.models.cnn import CNN_Mnist


def write_file(acc, loss, config, round):
    record = open(
        "{}_{}_{}_{}.txt".format(config.partition, config.sample_ratio,
                                 config.batch_size, config.epochs), "w")
    record.write(str(round) + "\n")
    record.write(str(config) + "\n")
    record.write(str(loss) + "\n")
    record.write(str(acc) + "\n")
    record.close()


# python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 5 --partition iid

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int, default=4000)

parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--epochs", type=int)
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
    data_indices = load_dict("mnist_noniid.pkl")
else:
    data_indices = load_dict("mnist_iid.pkl")

# fedlab setup
local_model = deepcopy(model)

trainer = SubsetSerialTrainer(model=local_model,
                              dataset=trainset,
                              data_slices=data_indices,
                              aggregator=aggregator,
                              args={
                                  "batch_size": args.batch_size,
                                  "epochs": args.epochs,
                                  "lr": args.lr
                              })

loss_ = []
acc_ = []

# train procedure
to_select = [i for i in range(total_client_num)]
for round in range(args.com_round):
    model_parameters = SerializationTool.serialize_model(model)
    selection = random.sample(to_select, num_per_round)
    aggregated_parameters = trainer.train(model_parameters=model_parameters,
                                          id_list=selection,
                                          aggregate=True)

    SerializationTool.deserialize_model(model, aggregated_parameters)

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    print("loss: {:.4f}, acc: {:.2f}".format(loss, acc))

    loss_.append(loss)
    acc_.append(acc)

    write_file(acc_, loss_, args, round + 1)

    if acc >= 0.985:
        break
