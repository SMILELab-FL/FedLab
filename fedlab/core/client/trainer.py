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


from ...utils.functional import AverageMeter, get_best_gpu
from ...utils.logger import Logger
from ...utils.serialization import SerializationTool
from ...utils.dataset.sampler import SubsetSampler


class ClientTrainer(ABC):
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
        self.cuda = cuda

        if self.cuda:
            # dynamic model assign
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
    def __init__(
        self,
        model,
        dataset,
        data_slices,
        aggregator=None,
        logger=None,
        cuda=True,
        args=None,
    ) -> None:

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
        batch_size = self.args.batch_size

        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[idx - 1],
                                  shuffle=True),
            batch_size=batch_size,
        )
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
        epochs, lr = self.args.epochs, self.args.lr

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


class SerialAsyncTrainer(SerialTrainer):
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
    def __init__(
        self,
        model,
        dataset,
        data_slices,
        aggregator=None,
        logger=None,
        cuda=True,
        args=None,
    ) -> None:

        super(SerialAsyncTrainer, self).__init__(model=model, dataset=dataset, data_slices=data_slices,
                                                 aggregator=aggregator, logger=logger, cuda=cuda, args=args)
        self.global_model_params = SerializationTool.serialize_model(self._model)
        self.current_time = 0
        # heap sort (aggregate_time, params, params_time)
        self.params_info = []

        self.stop_running = False
        # adapt alpha
        self.alpha = args.alpha
        self.strategy = args.strategy
        self.a = args.a
        self.b = args.b

    def train(self, id_list, cuda=True, aggregate=True):
        """Train local model with different dataset according to :attr:`idx` in :attr:`id_list`.

        Args:
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
        watch_aggregate = threading.Thread(target=self.watching_model_aggregate)
        watch_aggregate.start()

        aggregate_time_list = list(range(len(id_list)))
        random.shuffle(aggregate_time_list)

        for i, idx in enumerate(id_list):
            self._LOGGER.info(
                "starting training process of client [{}]".format(idx))

            data_loader = self._get_train_dataloader(idx=idx)

            self._train_alone(model_parameters=self.global_model_params,
                              train_loader=data_loader,
                              cuda=cuda)

            hp.heappush(self.params_info,
                        (aggregate_time_list[i], self.model_parameters, self.current_time))

        print("len:{}".format(len(self.params_info)))
        print("current_time:{}".format(self.current_time))

        # deal remaining update and close thread
        watch_aggregate.join(30)
        self.stop_running = False

        print("len:{}".format(len(self.params_info)))
        print("current_time:{}".format(self.current_time))
        return self.global_model_params

    def watching_model_aggregate(self):
        while self.stop_running is not True:
            if len(self.params_info) != 0 and self.current_time == self.params_info[0][0]:
                param_info = hp.heappop(self.params_info)  # (aggregate_time, params, params_time)
                alpha_T = self._adapt_alpha(receive_model_time=param_info[2])
                aggregated_params = self.aggregator(self.global_model_params, param_info[1], alpha_T)  # use aggregator
                self.global_model_params = aggregated_params
                self.current_time += 1

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
            return (staleness + 1) ** (-self.a)
        else:
            raise ValueError("Invalid strategy {}".format(self.strategy))


