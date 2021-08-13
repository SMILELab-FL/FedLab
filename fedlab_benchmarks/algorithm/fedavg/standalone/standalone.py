
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

from fedlab.core.client.trainer import SerialTrainer
from fedlab.utils.models.lenet import LeNet
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing


class mlp(nn.Module):
    def __init__(self):
        super(mlp,self).__init__()
        self.fc1 = nn.Linear(784,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5,5))
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.relu = nn. ReLU()
        self.fc2 = nn.Linear(512,10)
 
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int)

parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--partition", type=str)

parser.add_argument("--name", type=str)
parser.add_argument("--model", type=str)
# cuda config
parser.add_argument("--gpu", type=str, default="0,1,2,3")

args = parser.parse_args()



# get raw dataset
root = "../../../../../datasets/mnist/"
trainset = torchvision.datasets.MNIST(
    root=root, train=True, download=True, transform=transforms.ToTensor()
)
testset = torchvision.datasets.MNIST(
    root=root, train=False, download=True, transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset), drop_last=False, shuffle=False
)

# setup

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.model == "mlp":
    model = mlp().cuda()
elif args.model == "cnn":
    model = cnn().cuda()
else:
    raise ValueError("invalid model name ", args.model)



# FL settings
num_per_round = int(args.total_client*args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client  # client总数

if args.partition == "noniid":
    data_indices = noniid_slicing(
        trainset, num_clients=args.total_client, num_shards=200)
elif args.partition == "iid":
    data_indices = random_slicing(trainset, num_clients=args.total_client)
else:
    raise ValueError("invalid partition type ", args.partition)


# fedlab setup
local_model = deepcopy(model)

trainer = SerialTrainer(
    model=local_model, dataset=trainset, data_slices=data_indices, aggregator=aggregator, args=args
)
losses = []
acces = []


# train procedure
model_params = SerializationTool.serialize_model(model)
to_select = [i + 1 for i in range(total_client_num)]  # client_id 从1开始
for round in range(args.com_round):
    selection = random.sample(to_select, num_per_round)
    print(selection)
    aggregated_parameters = trainer.train(
        model_parameters=model_params, id_list=selection, aggregate=True
    )

    SerializationTool.deserialize_model(model, aggregated_parameters)
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    print("loss: {:.4f}, acc: {:.2f}".format(loss, acc))

    losses.append(loss)
    acces.append(acc)

    if (round+1) % 10 == 0:
        lossf = open("loss"+args.name+".txt", "w")
        lossf.write(str(losses))
        lossf.close()

        accf = open("accuracy"+args.name+".txt", "w")
        accf.write("round {}, sample ratio {}, lr {}, epoch {}, bs {}, partition {}".format(round, args.sample_ratio, args.lr, args.epochs, args.batch_size, args.partition))
        accf.write(str(acces))
        accf.close()
