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

from copy import deepcopy
import threading
import logging
from time import time

import torch

from fedlab_utils.logger import logger
from fedlab_utils.serialization import SerializationTool
from fedlab_utils.dataset.sampler import SubsetSampler
from fedlab_core.client.trainer import ClientTrainer
from fedlab_utils.aggregator import Aggregators


class ReturnThread(threading.Thread):
    def __init__(self, target, args):
        super(ReturnThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class SerialTrainer(ClientTrainer):
    """Train multiple clients with a single process or multiple threads.

    Args:
        model (torch.nn.Module): Model used in this federation.
        dataset (torch.nn.utils.dataset): local dataset for this group of clients.
        data_slices (list): subset of indices of dataset.
        aggregator (callable, :meth:`fedlab_utils.aggregator.Aggregators`): function to deal with a list of parameters.
        logger (:class:`fedlab_utils.logger`, optional): an util class to print log info to specific file and cmd line. If None, only cmd line. 
        cuda (bool): use GPUs or not.

    Notes:
        len(data_slices) == client_num, which means that every sub-indices of dataset represents a client's local dataset.

    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataset: torch.nn.utils.dataset,
                 data_slices: list,
                 aggregator: Aggregators,
                 logger: logger = None,
                 cuda: bool = True) -> None:

        super(SerialTrainer, self).__init__(model=model, cuda=cuda)

        self.dataset = dataset
        self.data_slices = data_slices  # [0,sim_client_num)
        self.client_num = len(data_slices)
        self.aggregator = aggregator

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def _get_dataloader(self, id, batch_size):
        """Return a dataloader used in :meth:`train`

        Args:
            client_id (int): client id to generate dataloader
            batch_size (int): batch size

        Returns:
            Dataloader for specific client sub-dataset
        """
        trainloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[id - 1],
                                  shuffle=True),
            batch_size=batch_size)
        return trainloader

    def _train_alone(self, id, model, epochs, data_loader, optimizer,
                     criterion, cuda):
        """single round of training

        Args:
            id (int): client id of this round.
            model (torch.nn.Module): model to be trained.
            epochs (int): the local epoch of training.
            data_loader (torch.utils.data.DataLoader): dataloader for data iteration.
            optimizer (torch.optim.Optimizer): Optimizer associated with model. Example, :class:`torch.nn.CrossEntropyLoss`.
            critereion (torch.nn.Loss): loss function. 
            cuda (bool): use GPUs or not.
        """
        model.train()
        for epoch in range(epochs):
            loss_sum = 0.0
            time_begin = time()
            for _, (data, target) in enumerate(data_loader):
                if cuda:
                    data = data.cuda()
                    target = target.cuda()

                output = self.model(data)

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.detach().item()

            print("Client[{}] Traning. Epoch {}/{}, Loss {:.4f}, Time {:.2f}s".
                  format(id, epoch + 1, epochs, loss_sum,
                         time() - time_begin))
        return SerializationTool.serialize_model(model)

    def train(self,
              model_parameters,
              epochs,
              lr,
              batch_size,
              id_list,
              cuda,
              multi_threading=False):
        """Train local model with different dataset according to id in id_list.

        Args:
            model_parameters (torch.Tensor): serialized model paremeters.
            epochs (int): number of epoch for local training.
            lr (float): learning rate
            batch_size (int): batch size
            id_list (list): client id in this train 
            cuda (bool): use GPUs or not.
            multi_threading (bool): use multiple threading to accelerate process.

        Returns:
            Merged serialized params

        #TODO: something wrong with multi_threading.
                异步训练失败？
        """
        param_list = []

        if multi_threading is True:
            threads = []

        for id in id_list:
            self._LOGGER.info(
                "starting training process of client [{}]".format(id))

            SerializationTool.deserialize_model(self._model, model_parameters)
            criterion = torch.nn.CrossEntropyLoss()
            data_loader = self._get_dataloader(id=id, batch_size=batch_size)

            if multi_threading is False:

                optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
                self._train_alone(id,
                                  self._model,
                                  epochs=epochs,
                                  data_loader=data_loader,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  cuda=cuda)
                param_list.append(SerializationTool.serialize_model(
                    self.model))

            else:

                model = deepcopy(self._model)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                args = (id, model, epochs, data_loader, optimizer, criterion,
                        cuda)
                t = ReturnThread(target=self._train_alone, args=args)
                t.start()
                threads.append(t)

        if multi_threading is True:
            for t in threads:
                t.join()
            for t in threads:
                param_list.append(t.get_result())

        # aggregate model parameters
        return self.aggregator(param_list)
