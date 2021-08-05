# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import sys

sys.path.append("../")
from copy import deepcopy

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from fedlab.utils.dataset.sampler import FedDistributedSampler
from fedlab.core.client.trainer import ClientSGDTrainer
from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.manager import ServerSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.utils.models.lenet import LeNet


class TestServer:
    def __init__(self, ip, port, world_size, model) -> None:

        model = deepcopy(model)

        ps = SyncParameterServerHandler(model,
                                        client_num_in_total=world_size - 1)

        network = DistNetwork(address=(ip, port),
                              world_size=world_size,
                              rank=0)

        self.p = ServerSynchronousManager(handler=ps, network=network)

class TestClient:
    def __init__(self, ip, port, world_size, rank, model, trainset) -> None:
        model = deepcopy(model)
        lr = 0.01
        momentum = 0.9
        criterion = nn.CrossEntropyLoss()

        trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=FedDistributedSampler(trainset,
                                          client_id=rank,
                                          num_replicas=world_size - 1))

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=momentum)

        network = DistNetwork(address=(ip, port),
                              world_size=world_size,
                              rank=rank)

        handler = ClientSGDTrainer(model,
                                   trainloader,
                                   epoch=1,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   cuda=True)

        self.p = ClientPassiveManager(handler=handler, network=network)

class FedAvgTestCase(unittest.TestCase):
    def setUp(self) -> None:
        ip = "127.0.0.1"
        port = "12345"
        world_size = 3
        model = LeNet()
        trainset = torchvision.datasets.MNIST(root="./data/mnist/",
                                              train=True,
                                              download=True,
                                              transform=transforms.ToTensor())

        self.server = TestServer(ip=ip,
                                 port=port,
                                 world_size=world_size,
                                 model=model)

        self.client1 = TestClient(ip=ip,
                                  port=port,
                                  world_size=world_size,
                                  rank=1,
                                  model=model,
                                  trainset=trainset)
        self.client2 = TestClient(ip=ip,
                                  port=port,
                                  world_size=world_size,
                                  rank=2,
                                  model=model,
                                  trainset=trainset)

    def tearDown(self) -> None:
        return super().tearDown()

    @unittest.skipUnless(torch.cuda.is_available(), "torch.cuda is required")
    def test_fedavg(self):

        self.server.p.start()
        self.client1.p.start()
        self.client2.p.start()

        self.server.p.join()
        self.client1.p.join()
        self.client2.p.join()
