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

from ...utils.logger import Logger
from ...utils.serialization import SerializationTool
from ...utils.dataset.sampler import SubsetSampler
from ...utils.aggregator import Aggregators

from .trainer import ClientTrainer


# TODO: PUT `SerialTrainer` into trainer.py file
class SerialTrainer(ClientTrainer):
    """Train multiple clients in a single process.

    Args:
        model (torch.nn.Module): Model used in this federation.
        dataset (torch.utils.data.Dataset): Local dataset for this group of clients.
        data_slices (list[list]): subset of indices of dataset.
        aggregator (Aggregators, callable, optional): Function to perform aggregation on a list of model parameters.
        logger (Logger, optional): Logger for the current trainer. If ``None``, only log to command line.
        cuda (bool): Use GPUs or not. Default: ``True``.

    Notes:
        ``len(data_slices) == client_num``, that is, each sub-index of :attr:`dataset` corresponds to a client's local dataset one-by-one.

    """

    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 aggregator=None,
                 logger=None,
                 cuda=True) -> None:

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

    def _get_train_dataloader(self, idx):
        """Return a training dataloader used in :meth:`train` for client with :attr:`id`

        Args:
            idx (int): :attr:`idx` of client to generate dataloader

        Note:
            :attr:`idx` here is not equal to ``client_id`` in the FL setting. It is the index of client in current :class:`SerialTrainer`.

        Returns:
            :class:`DataLoader` for specific client sub-dataset
        """
        batch_size = 128

        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[idx - 1],
                                  shuffle=True),
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
        SerializationTool.deserialize_model(self._model, model_parameters)
        epochs = 5
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=0.1)
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

        if aggregate is True:
            # aggregate model parameters of this client group
            return self.aggregator(param_list)
        else:
            return param_list
