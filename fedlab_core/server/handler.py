import logging
import os
import random
import torch
import copy
from queue import Queue

from abc import ABC, abstractmethod
from fedlab_utils.message_code import MessageCode
from fedlab_utils.serialization import SerializationTool
from fedlab_utils.aggregator import Aggregators


class ParameterServerBackendHandler(ABC):
    """An abstract class representing handler for parameter server.

    Please make sure that you self-defined server handler class subclasses this class

    Example:
        read sourcecode of :class:`SyncSGDParameterServerHandler` below
    """
    def __init__(self, model, cuda=False) -> None:
        self.cuda = cuda
        if cuda:
            self._model = model.cuda()
        else:
            self._model = model.cpu()

    @abstractmethod
    def update_model(self, serialized_params_list):
        """Override this function to update global model

        Args:
            serialized_params_list (list[torch.Tensor]): a list of serialized model parameters collected from different clients
        """
        raise NotImplementedError()

    @property
    def model(self):
        return self._model


class SyncParameterServerHandler(ParameterServerBackendHandler):
    """Synchronous Parameter Server Handler

    Backend of synchronous parameter server: this class is responsible for backend computing.

    Synchronous parameter server will wait for every client to finish local training process before the
    next FL round.

    details in paper: http://proceedings.mlr.press/v54/mcmahan17a.html

    Args:
        model (torch.nn.Module): Model used in this federation
        client_num_in_total (int): Total number of clients in this federation
        cuda (bool): Use GPUs or not. Default: ``False``
        select_ratio (float): ``select_ratio * client_num`` is the number of clients to join every FL round. Default: ``1.0``
        logger (:class:`fedlab_utils.logger`, optional): Tools, used to output information.
    """
    def __init__(self,
                 model,
                 client_num_in_total,
                 cuda=False,
                 select_ratio=1.0,
                 logger=None):
        super(SyncParameterServerHandler, self).__init__(model, cuda)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        if select_ratio < 0.0 or select_ratio > 1.0:
            raise ValueError("Invalid select ratio: {}".format(select_ratio))

        if client_num_in_total < 1:
            raise ValueError(
                "Invalid total client number: {}".format(client_num_in_total))

        self.client_num_in_total = client_num_in_total
        self.select_ratio = select_ratio
        self.client_num_per_round = max(
            1, int(self.select_ratio * self.client_num_in_total))

        # client buffer
        self.client_buffer_cache = {}
        self.cache_cnt = 0

    def select_clients(self):
        """Return a list of client rank indices selected randomly"""
        id_list = [i + 1 for i in range(self.client_num_in_total)]
        selection = random.sample(id_list, self.client_num_per_round)
        return selection

    def add_single_model(self, sender_rank, serialized_params):
        """Deal with incoming model parameters

        Args:
            sender_rank (int): rank of sender in distributed
            serialized_params (torch.Tensor): serialized model parameters
        """
        if self.client_buffer_cache.get(sender_rank) is not None:
            self._LOGGER.info(
                "parameters from {} has existed".format(sender_rank))
            return

        self.cache_cnt += 1
        self.client_buffer_cache[sender_rank] = serialized_params.clone()

        if self.cache_cnt == self.client_num_per_round:
            self.update_model(list(self.client_buffer_cache.values()))
            return True
        else:
            return False

    def update_model(self, serialized_params_list):
        """update global model"""
        # use aggregator
        serialized_parameters = Aggregators.fedavg_aggregate(
            serialized_params_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

        # reset
        self.cache_cnt = 0
        self.client_buffer_cache = {}
        self.train_flag = False

    def train(self):
        self.train_flag = True


class AsyncParameterServerHandler(ParameterServerBackendHandler):
    """Asynchronous ParameterServer Handler

    Update global model immediately after receiving a ParameterUpdate message
    paper: https://arxiv.org/abs/1903.03934

    Args:
        model (torch.nn.Module): Global model in server
        client_num_in_total (int): the num of client in federation.
        cuda (bool): Use GPUs or not.
        logger (:class:`fedlab_utils.logger`, optional): Tools, used to output information.
    """
    def __init__(self, model, client_num_in_total, cuda=False, logger=None):
        super(AsyncParameterServerHandler, self).__init__(model, cuda)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.alpha = 0.5
        self.client_num_in_total = client_num_in_total
        self.model_update_time = torch.zeros(1)  # record the current model's updated time

    def update_model(self, model_parameters, model_time):
        """"update global model from client_model_queue"""
        latest_serialized_parameters = SerializationTool.serialize_model(
            self.model)
        merged_params = Aggregators.fedasgd_aggregate(
            latest_serialized_parameters, model_parameters,
            self.alpha)  # use aggregator
        SerializationTool.deserialize_model(self._model, merged_params)
        self.model_update_time += 1

    def adapt_alpha(self, receive_model_time):
        """update the alpha according to staleness"""
        self.alpha = torch.mul(self.alpha, 1)
