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

from ..client import ORDINARY_TRAINER
from ...utils.functional import AverageMeter, get_best_gpu
from ...utils.logger import Logger
from ...utils.serialization import SerializationTool
from ...utils.dataset.sampler import SubsetSampler
from ..model_maintainer import ModelMaintainer

class ClientTrainer(ModelMaintainer):
    """An abstract class representing a client backend handler.

    In our framework, we define the backend of client handler show manage its local model.
    It should have a function to update its model called :meth:`train`.

    If you use our framework to define the activities of client, please make sure that your self-defined class
    should subclass it. All subclasses should overwrite :meth:`train`.

    Args:
        model (torch.nn.Module): Model used in this federation
        cuda (bool): Use GPUs or not
    """
    def __init__(self, model, cuda):
        super().__init__(model, cuda)
        
        self.client_num = 1  # default is 1.
        self.type = ORDINARY_TRAINER

        if self.cuda:
            # dynamic gpu acquire.
            self.gpu = get_best_gpu()
            self._model = model.cuda(self.gpu)
        else:
            self._model = model.cpu()

    @abstractmethod
    def train(self):
        """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
        raise NotImplementedError()

    @property
    def model(self):
        """attribute"""
        return self._model

    @property
    def model_parameters(self):
        """attribute"""
        return SerializationTool.serialize_model(self._model)
    
    @property
    def shape_list(self):
        """attribute"""
        shape_list = []
        for parameters in self._model.parameters():
            shape_list.append(parameters.shape)
        return shape_list

class ClientSGDTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): model used in federation.
        data_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        epochs (int): the number of local epoch.
        optimizer (torch.optim.Optimizer, optional): optimizer for this client's model. If set to ``None``, will use :func:`torch.optim.SGD` with :attr:`lr` of 0.1 and :attr:`momentum` of 0.9 as default.
        criterion (torch.nn.Loss, optional): loss function used in local training process. If set to ``None``, will use:func:`nn.CrossEntropyLoss` as default.
        cuda (bool, optional): use GPUs or not. Default: ``True``.
        logger (Logger, optional): :attr:`logger` for client trainer. . If set to ``None``, none logging output files will be generated while only on screen. Default: ``None``.
    """
    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=True,
                 logger=None):
        super(ClientSGDTrainer, self).__init__(model, cuda)

        self._data_loader = data_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def train(self, model_parameters, epochs=None) -> None:
        """
        Client trains its local model on local dataset.åß

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            epochs (int): Number of epoch for current local training.
        """
        self._LOGGER.info("Local train procedure is started")
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        if epochs is None:
            epochs = self.epochs
        self._LOGGER.info("Local train procedure is running")
        for _ in range(epochs):
            self._model.train()
            for inputs, labels in self._data_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(
                        self.gpu)

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")
        return self.model_parameters
