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

import time
from abc import ABC, abstractmethod
import logging

import torch
from torch import nn
import threading
import heapq as hp
import random

from ..trainer import ClientTrainer
from ....utils.functional import AverageMeter, get_best_gpu
from ....utils.logger import Logger
from ....utils.serialization import SerializationTool
from ....utils.dataset.sampler import SubsetSampler


class SerialTrainer(ClientTrainer):
    """Train multiple clients in a single process.
    """
    def __init__(self, model, cuda):
        super().__init__(model, cuda)

    def _train_alone(self):
        raise NotImplementedError()

    def _get_dataloader(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()


class SubsetSerialTrainer(SerialTrainer):
    """Train multiple clients in a single process.

    Args:
        model (torch.nn.Module): Model used in this federation.
        dataset (torch.utils.data.Dataset): Local dataset for this group of clients.
        data_slices (list[list]): subset of indices of dataset.
        aggregator (Aggregators, callable, optional): Function to perform aggregation on a list of model parameters.
        logger (Logger, optional): Logger for the current trainer. If ``None``, only log to command line.
        cuda (bool): Use GPUs or not. Default: ``True``.
        args (dict, optional): Uncertain variables.
    Notes:
        ``len(data_slices) == client_num``, that is, each sub-index of :attr:`dataset` corresponds to a client's local dataset one-by-one.

    """
    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 aggregator=None,
                 logger=None,
                 cuda=True,
                 args=None) -> None:

        super(SerialTrainer, self).__init__(model=model, cuda=cuda)

        self.dataset = dataset
        self.data_slices = data_slices  # [0, client_num)
        self.client_num = len(data_slices)
        self.aggregator = aggregator

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.args = args

    def _get_train_dataloader(self, idx):
        """Return a training dataloader used in :meth:`train` for client with :attr:`id`

        Args:
            idx (int): :attr:`idx` of client to generate dataloader

        Note:
            :attr:`idx` here is not equal to ``client_id`` in the FL setting. It is the index of client in current :class:`SerialTrainer`.

        Returns:
            :class:`DataLoader` for specific client sub-dataset
        """
        batch_size = self.args["batch_size"]

        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[idx], shuffle=True),
            batch_size=batch_size)
        return train_loader

    def _train_alone(self, model_parameters, train_loader, cuda):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): model parameters.
            train_loader (torch.utils.data.DataLoader): dataloader for data iteration.
            cuda (bool): use GPUs or not.
        """
        epochs, lr = self.args["epochs"], self.args["lr"]
        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._model.train()

        for _ in range(epochs):
            for data, target in train_loader:
                if cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                output = self.model(data)

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model_parameters

    def train(self, model_parameters, id_list, cuda=True, aggregate=True):
        """Train local model with different dataset according to :attr:`idx` in :attr:`id_list`.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            id_list (list[int]): Client id in this training serial.
            cuda (bool): Use GPUs or not. Default: ``True``.
            aggregate (bool): Whether to perform partial aggregation on this group of clients' local model in the end of local training round.

        Note:
            Normally, aggregation is performed by server, while we provide :attr:`aggregate` option here to perform
            partial aggregation on current client group. This partial aggregation can reduce the aggregation workload
            of server.

        Returns:
            Serialized model parameters / list of model parameters.
        """
        param_list = []
        for idx in id_list:
            self._LOGGER.info(
                "starting training process of client [{}]".format(idx))

            data_loader = self._get_train_dataloader(idx=idx)
            self._train_alone(model_parameters=model_parameters,
                              train_loader=data_loader,
                              cuda=cuda)
            param_list.append(self.model_parameters)

        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list)
            return aggregated_parameters
        else:
            return param_list


class AsyncSerialTrainer(SerialTrainer):
    """Train multiple clients in a single process.

        Args:
            model (torch.nn.Module): Model used in this federation.
            dataset (torch.utils.data.Dataset): Local dataset for this group of clients.
            data_slices (list[list]): subset of indices of dataset.
            aggregator (Aggregators, callable, optional): Function to perform aggregation on a list of model parameters.
            logger (Logger, optional): Logger for the current trainer. If ``None``, only log to command line.
            cuda (bool): Use GPUs or not. Default: ``True``.
            args (dict, optional): Uncertain variables.
        Notes:
            ``len(data_slices) == client_num``, that is, each sub-index of :attr:`dataset` corresponds to a client's local dataset one-by-one.

        """
    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 aggregator=None,
                 logger=None,
                 cuda=True,
                 args=None) -> None:

        super(SerialAsyncTrainer, self).__init__(model=model,
                                                 dataset=dataset,
                                                 data_slices=data_slices,
                                                 aggregator=aggregator,
                                                 logger=logger,
                                                 cuda=cuda,
                                                 args=args)

    def _train_alone(self, model_parameters, train_loader, cuda):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): model parameters.
            train_loader (torch.utils.data.DataLoader): dataloader for data iteration.
            cuda (bool): use GPUs or not.
        """
        epochs, lr = self.args["epochs"], self.args["lr"]

        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._model.train()

        for _ in range(epochs):
            for data, target in train_loader:
                if cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                output = self.model(data)

                l2_reg = self._l2_reg_params(
                    global_model_parameters=model_parameters,
                    reg_lambda=self.args["reg_lambda"],
                    reg_condition=0)
                if l2_reg is not None:
                    loss = criterion(output,
                                     target) + l2_reg * self.args["reg_lambda"]
                else:
                    loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model_parameters

    def _l2_reg_params(self, global_model_parameters, reg_lambda,
                       reg_condition):
        l2_reg = None
        if reg_lambda <= reg_condition:
            return l2_reg

        current_index = 0
        for parameter in self.model.parameters():
            parameter = parameter.cpu()
            numel = parameter.data.numel()
            size = parameter.data.size()
            global_parameter = global_model_parameters[
                current_index:current_index + numel].view(size)
            current_index += numel
            if l2_reg is None:
                l2_reg = (parameter - global_parameter).norm(2)
            else:
                l2_reg = l2_reg + (parameter - global_parameter).norm(2)

        return l2_reg * reg_lambda
