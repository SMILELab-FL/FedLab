import sys
import torch
sys.path.append('/home/zengdun/FedLab/')
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import random
import os 
import argparse
import wandb
wandb.init(project='fedavg', entity='zengdun')

# fedlab modules
from fedlab_core.client.serial_handler import SerialHandler
from models.lenet import LeNet
from fedlab_utils.aggregator import Aggregators
from fedlab_utils.serialization import SerializationTool
from fedlab_utils.functional import evaluate
from fedlab_utils.dataset.slicing import noniid_slicing


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
parser.add_argument('--cuda', type=str, default=0)
args = parser.parse_args()

config = wandb.config
config.learning_rate = 0.1
config.batch_size = 128
config.epochs = 10

config.communicate_round = args.com_round

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

model = LeNet().cuda()
aggregator = Aggregators.fedavg_aggregate
criterion = torch.nn.CrossEntropyLoss()

trainset = torchvision.datasets.MNIST(
    root='/home/zengdun/datasets/mnist/', train=True, download=True, transform=transforms.ToTensor())

testset = torchvision.datasets.MNIST(
    root='/home/zengdun/datasets/mnist/', train=False, download=True, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), drop_last=False, shuffle=False)

total_client_num = args.total_client  # client总数
data_indices = noniid_slicing(trainset, num_clients=args.total_client, num_shards=200)

test_handler = SerialHandler(local_model=deepcopy(model), aggregator=aggregator, dataset=trainset, sim_client_num=total_client_num, client_data_indices=data_indices)
model_params = SerializationTool.serialize_model(model)
to_select = [i+1 for i in range(total_client_num)] # client_id 从1开始

for round in range(args.com_round):
    selection = random.sample(to_select, args.num_per_round)
    print(selection)
    model_params = test_handler.train(epochs=args.epochs, batch_size=args.batch_size, idx_list=selection, model_parameters=model_params, cuda=True)
    SerializationTool.deserialize_model(model, model_params)
    loss, acc = evaluate(model, criterion, test_loader, cuda=True)
    wandb.log({"Test Accuracy": acc})