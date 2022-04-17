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

sys.path.append("../../")
torch.manual_seed(0)

from fedlab.core.client.serial_trainer import SubsetSerialTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.utils.functional import get_best_gpu, load_dict

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int)

parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--cuda", type=bool, default=False)

args = parser.parse_args()


# torch model
class MLP(nn.Module):

    def __init__(self, input_size=784, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# get mnist dataset
root = "../../tests/data/mnist/"
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

if args.cuda:
    gpu = get_best_gpu()
    model = MLP().cuda(gpu)
else:
    model = MLP()

# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client  # client总数

data_indices = load_dict("mnist_partition.pkl")

# fedlab setup
trainer = SubsetSerialTrainer(model=model,
                              dataset=trainset,
                              data_slices=data_indices,
                              args={
                                  "batch_size": args.batch_size,
                                  "epochs": args.epochs,
                                  "lr": args.lr
                              })

# train procedure
to_select = [i for i in range(total_client_num)]
for round in range(args.com_round):
    model_parameters = SerializationTool.serialize_model(model)
    selection = random.sample(to_select, num_per_round)
    parameters_list = trainer.local_process(payload=[model_parameters],
                                            id_list=selection)

    SerializationTool.deserialize_model(model, aggregator(parameters_list))

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    print("loss: {:.4f}, acc: {:.2f}".format(loss, acc))
