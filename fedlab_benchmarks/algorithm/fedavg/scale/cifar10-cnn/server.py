import sys
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
torch.manual_seed(0)
sys.path.append('../../../../../')

from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import AverageMeter

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(model, criterion, test_loader):
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.sum, acc_.avg


def write_file(acces, losses, name="cifar10"):
    print("wtring")
    record = open(name + ".txt", "w")

    record.write(str(losses) + "\n\n")
    record.write(str(acces) + "\n\n")
    record.close()


class RecodeHandler(SyncParameterServerHandler):
    def __init__(self,
                 model,
                 client_num_in_total,
                 test_loader,
                 global_round=5,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=None):
        super().__init__(model,
                         client_num_in_total,
                         global_round=global_round,
                         cuda=cuda,
                         sample_ratio=sample_ratio,
                         logger=logger)

        self.test_loader = test_loader
        self.loss_ = []
        self.acc_ = []

    def _update_model(self, model_parameters_list):
        super()._update_model(model_parameters_list)

        loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                             self.test_loader)

        self.loss_.append(loss)
        self.acc_.append(acc)

        write_file(self.acc_, self.loss_)


# python server.py --world_size 11
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL server example')

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=str, default="3002")
    parser.add_argument('--world_size', type=int)

    parser.add_argument('--round', type=int, default=100)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=0.1)

    args = parser.parse_args()

    #model = torchvision.models.resnet34()
    model = Net()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='../../../../datasets/data/cifar10/',
        train=False,
        download=True,
        transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=int(len(testset) / 10),
                                             drop_last=False,
                                             shuffle=False)

    handler = RecodeHandler(model,
                            client_num_in_total=1,
                            global_round=args.round,
                            sample_ratio=args.sample,
                            test_loader=testloader,
                            cuda=True)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)

    manager_ = ScaleSynchronousManager(network=network, handler=handler)
    manager_.run()
