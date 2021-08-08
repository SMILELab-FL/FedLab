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

from ...utils.logger import logger
from ...utils.serialization import SerializationTool
from ...utils.dataset.sampler import SubsetSampler
from ...utils.aggregator import Aggregators

from .trainer import ClientTrainer


class SerialTrainer(ClientTrainer):
    """Train multiple clients with a single process or multiple threads.

    Args:
        model (torch.nn.Module): Model used in this federation.
        dataset (torch.utils.data.Dataset): local dataset for this group of clients.
        data_slices (list): subset of indices of dataset.
        aggregator (Aggregators, callable, optional): function to aggregate a list of parameters.
        logger (logger, optional): an util class to print log info to specific file and cmd line. If None, only cmd line. 
        cuda (bool): use GPUs or not.

    Notes:
        len(data_slices) == client_num, which means that every sub-indices of dataset represents a client's local dataset.

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
        self.data_slices = data_slices  # [0,sim_client_num)
        self.client_num = len(data_slices)
        self.aggregator = aggregator

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def _get_dataloader(self, id):
        """Return a dataloader used in :meth:`train`

        Args:
            id (int): client id to generate dataloader
            batch_size (int): batch size

        Returns:
            Dataloader for specific client sub-dataset
        """
        batch_size = 128

        trainloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[id - 1],
                                  shuffle=True),
            batch_size=batch_size)
        return trainloader

    def _train_alone(self, model_parameters, data_loader, cuda):
        """single round of training

        Note:
            Overwrite this method to customize the PyTorch traning pipeline.

        Args:
            id (int): client id of this round.
            model_parameters (torch.Tensor): model parameters.
            data_loader (torch.utils.data.DataLoader): dataloader for data iteration.
            cuda (bool): use GPUs or not.
        """
        SerializationTool.deserialize_model(self._model, model_parameters)
        epochs = 5
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=0.1)
        self._model.train()

        for _ in range(epochs):

            for data, target in data_loader:
                if cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                output = self.model(data)

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model_parameters

    def train(self, model_parameters, id_list, cuda, aggregate=True):
        """Train local model with different dataset according to id in id_list.

        Args:
            model_parameters (torch.Tensor): serialized model paremeters.
            id_list (list): client id in this train.
            cuda (bool): use GPUs or not.

        Returns:
            serialized model parameters / list of model parameters.
        """
        param_list = []

        for id in id_list:
            self._LOGGER.info(
                "starting training process of client [{}]".format(id))

            data_loader = self._get_dataloader(id=id)

            self._train_alone(model_paramters=model_parameters,
                              data_loader=data_loader,
                              cuda=cuda)

            param_list.append(self.model)

        if aggregate is True:
            # aggregate model parameters
            return self.aggregator(param_list)
        else:
            return param_list
