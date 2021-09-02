import os
import argparse
import random
from copy import deepcopy
import torchvision.transforms as transforms
from torch import nn
import torchvision
import torch

torch.manual_seed(0)
from fedlab.core.client.scale.trainer import AsyncSerialTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, load_dict
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

import sys

sys.path.append("../../../../")
from fedlab_benchmarks.models.cnn import AlexNet_CIFAR10

import heapq as hp
import threading

# python standalone.py --com_round 2000 --sample_ratio 0.05 --batch_size 100 --epochs 5 --partition iid --name test1 --lr 0.01 --alpha 0.6
# python standalone.py --com_round 2000 --sample_ratio 0.05 --batch_size 100 --epochs 5 --partition noniid --name test2 --lr 0.01 --alpha 0.6


def write_file(acc, loss, args, round):
    record = open("exp_" + args.partition + "_" + args.name + ".txt", "w")
    record.write(
        "current {}, sample ratio {}, lr {}, epoch {}, bs {}, partition {}\n\n"
        .format(round + 1, args.sample_ratio, args.lr, args.epochs,
                args.batch_size, args.partition))
    record.write(str(acc) + "\n\n")
    record.write(str(loss) + "\n\n")
    record.close()


class AsyncAggregate:
    """aggregate asynchronously server util

    Args:
        model_parameters (torch.Tensor): model parameters.
        aggregator (Aggregators, callable, optional): Function to perform aggregation on a list of model parameters.
        alpha (float): mixing hyperparameter in FedAsync algorithm, range (0,1)
        strategy (str): strategy for weighting function, values ``constant``, ``hinge`` and ``polynomial``
        a (int): parameter for ``hinge`` and ``polynomial`` strategy
        b (int): parameter for ``hinge`` strategy
    """
    def __init__(self, model_parameters, aggregator, alpha, strategy, a, b,
                 max_staleness):
        self.model_parameters = model_parameters
        self.aggregator = aggregator
        self.alpha = alpha
        self.strategy = strategy
        self.a = a
        self.b = b
        self.current_time = 0
        self.param_counter = 0
        self.max_staleness = max_staleness
        # each (model_time+staleness, param_counter, model_param, model_time)
        self.params_info_hp = []

    def model_aggregate(self):
        while len(self.params_info_hp) > 0:
            if self.current_time > self.params_info_hp[0][0]:
                # remove old aggregate_time, which has been implemented
                hp.heappop(self.params_info_hp)
                return False
            elif self.current_time == self.params_info_hp[0][0]:
                param_info = hp.heappop(
                    self.params_info_hp
                )  # (model_time+staleness, counter, model_param, model_time)
                # solve same aggregate_time(model_time+staleness) conflict question, drop remaining same
                while len(self.params_info_hp) != 0 and param_info[
                        0] == self.params_info_hp[0][0]:
                    hp.heappop(self.params_info_hp)

                alpha_T = self._adapt_alpha(receive_model_time=param_info[3])
                aggregated_params = self.aggregator(self.model_parameters,
                                                    param_info[2],
                                                    alpha_T)  # use aggregator
                self.model_parameters = aggregated_params
                self.current_time += 1
                return True
            else:
                return False

    def _adapt_alpha(self, receive_model_time):
        """update the alpha according to staleness"""
        staleness = self.current_time - receive_model_time
        if self.strategy == "constant":
            return torch.mul(self.alpha, 1)
        elif self.strategy == "hinge" and self.b is not None and self.a is not None:
            if staleness <= self.b:
                return torch.mul(self.alpha, 1)
            else:
                return torch.mul(self.alpha,
                                 1 / (self.a * ((staleness - self.b) + 1)))
        elif self.strategy == "polynomial" and self.a is not None:
            return (staleness + 1)**(-self.a)
        else:
            raise ValueError("Invalid strategy {}".format(self.strategy))

    def append_params_to_hp(self, params_list):
        current_time = self.current_time
        for param in params_list:
            staleness = random.randint(0, self.max_staleness)
            hp.heappush(self.params_info_hp,
                        (staleness + current_time, self.param_counter, param,
                         current_time))
            self.param_counter += 1


# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int, default=5000)

parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--partition", type=str)

parser.add_argument("--name", type=str)
# async update config
parser.add_argument("--alpha", type=float, default=0.6)
parser.add_argument("--strategy", type=str, default='constant')
parser.add_argument("--a", type=int, default=10)
parser.add_argument("--b", type=int, default=4)
parser.add_argument("--reg_lambda", type=float, default=0.005)
parser.add_argument("--max_staleness", type=int, default=4)

args = parser.parse_args()

# get raw dataset
root = '../../../datasets/data/cifar10/'
trainset = torchvision.datasets.CIFAR10(root=root,
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root=root,
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=len(testset),
                                          drop_last=False,
                                          shuffle=False)

# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model = AlexNet_CIFAR10()

# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedasync_aggregate
total_client_num = args.total_client  # client总数

if args.partition == "noniid":
    data_indices = load_dict("cifar10_noniid.pkl")
elif args.partition == "iid":
    data_indices = load_dict("cifar10_iid.pkl")
else:
    raise ValueError("invalid partition type ", args.partition)

# fedlab setup
local_model = deepcopy(model)

args_test = {
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "reg_lambda": args.reg_lambda
}

trainer = AsyncSerialTrainer(
    model=local_model,
    dataset=trainset,
    data_slices=data_indices,
    aggregator=aggregator,
    args=args_test,
)

loss_ = []
acc_ = []

# train procedure

to_select = [i for i in range(total_client_num)]  # client_id 从1开始

async_aggregate = AsyncAggregate(
    model_parameters=SerializationTool.serialize_model(model),
    aggregator=aggregator,
    alpha=args.alpha,
    strategy=args.strategy,
    a=args.a,
    b=args.b,
    max_staleness=args.max_staleness)

for round in range(args.com_round):
    model_parameters = async_aggregate.model_parameters
    selection = random.sample(to_select, num_per_round)
    print(selection)
    params_list = trainer.train(model_parameters=model_parameters,
                                id_list=selection,
                                aggregate=False)

    async_aggregate.append_params_to_hp(params_list)

    if async_aggregate.model_aggregate():

        SerializationTool.deserialize_model(model,
                                            async_aggregate.model_parameters)

        criterion = nn.CrossEntropyLoss()
        loss, acc = evaluate(model, criterion, test_loader)

        loss_.append(loss)
        acc_.append(acc)
        write_file(acc_, loss_, args, round)