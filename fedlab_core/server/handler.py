import os
import random
import torch
import copy
from queue import Queue

from abc import ABC, abstractmethod
from fedlab_utils.logger import logger
from fedlab_utils.message_code import MessageCode
from fedlab_utils.serialization import SerializationTool


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
    def on_receive(self):
        """Override this function to define what the server to do when receiving message from client"""
        raise NotImplementedError()

    @abstractmethod
    def add_single_model(self, sender_rank, serialized_params):
        """Override this function to deal with incoming model

        Args:
            sender_rank (int): rank of sender in distributed
            serialized_params (torch.Tensor): serialized model parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def update_model(self, serialized_params_list):
        """Override this function to update global model

        Args:
            serialized_params_list (list[torch.Tensor]): a list of serialized model parameters collected from different
        clients
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
        select_ratio (float): ``select_ratio * client_num`` is the number of clients to join every FL round. Default:
    ``1.0``
        logger_file (str, optional): Path to the log file for client handler. Default: ``"server_handler.txt"``
        logger_name (str, optional): Class name to initialize logger for client handler. Default: ``"server handler"``
    """
    def __init__(self,
                 model,
                 client_num_in_total,
                 cuda=False,
                 select_ratio=1.0,
                 logger_path="server_handler",
                 logger_name="server handler"):
        super(SyncParameterServerHandler, self).__init__(model, cuda)

        if select_ratio < 0.0 or select_ratio > 1.0:
            raise ValueError("Invalid select ratio: {}".format(select_ratio))

        if client_num_in_total < 1:
            raise ValueError(
                "Invalid total client number: {}".format(client_num_in_total))

        self.client_num_in_total = client_num_in_total
        self.select_ratio = select_ratio
        self.client_num_per_round = max(
            1, int(self.select_ratio * self.client_num_in_total))

        self._LOGGER = logger(os.path.join("log", logger_path + ".txt"),
                              logger_name)

        # client buffer
        self.client_buffer_cache = {}
        self.cache_cnt = 0

        # setup
        self.train_flag = False

    def on_receive(self, sender_rank, message_code, serialized_params) -> None:
        """Define what parameter server does when receiving a single client's message

        Args:
            sender_rank (int): Rank of client process sending this local model
            message_code (MessageCode): Agreements code defined in :class:`MessageCode`
            serialized_params (torch.Tensor): Serialized local model parameters from client
        """

        self._LOGGER.info("Processing message: {} from rank {}".format(
            message_code.name, int(sender_rank)))

        if message_code == MessageCode.ParameterUpdate:
            # update model parameters
            self.add_single_model(sender_rank, serialized_params)
            # update server model when client_buffer_cache is full
            if self.cache_cnt == self.client_num_per_round:
                self.update_model(list(self.client_buffer_cache.values()))
        else:
            raise Exception("Undefined message type!")

    def select_clients(self):
        """Return a list of client rank indices selected randomly"""
        id_list = [i + 1 for i in range(self.client_num_in_total)]
        selection = random.sample(id_list, self.client_num_per_round)
        return selection

    def add_single_model(self, sender_rank, serialized_params):
        """deal with single model's parameters"""
        if self.client_buffer_cache.get(sender_rank) is not None:
            self._LOGGER.info(
                "parameters from {} has existed".format(sender_rank))
            return

        self.cache_cnt += 1
        self.client_buffer_cache[sender_rank] = serialized_params.clone()

    def update_model(self, serialized_params_list):
        """update global model"""
        serialized_parameters = torch.mean(torch.stack(serialized_params_list),
                                           dim=0)
        SerializationTool.restore_model(self._model, serialized_parameters)

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
        cuda (bool): Use GPUs or not
    """

    # TODO: unfinished
    def __init__(self,
                 model,
                 client_num_in_total,
                 cuda=False,
                 logger_path="server_handler",
                 logger_name="server handler"):
        super(AsyncParameterServerHandler, self).__init__(model, cuda)

        self.alpha = 0.5
        self.client_num_in_total = client_num_in_total
        self.client_num_per_round = 2  # test

        # package: [model, T]
        # need a Queue

        self.model_update_time = 0  # record the current model's updated time
        # need a Queue to receive the updated model from each client, not useful?
        self.client_model_queue = Queue()

        self._LOGGER = logger(os.path.join("log", logger_path + ".txt"),
                              logger_name)

    def on_receive(self, sender_rank, message_code, content_list):
        """Define what parameter server does when receiving a single client's message
        """

        self._LOGGER.info(
            "Processing message: {} from rank {}, the received updated model time is {}, "
            "the handle's current model time is {}".format(
                message_code.name, int(sender_rank),
                int(content_list[1].item()), self.model_update_time))

        if message_code == MessageCode.ParameterUpdate:
            # update local model parameters, and update server model async
            self.add_single_model(sender_rank, content_list)
            self.update_model()

        elif message_code == MessageCode.GradientUpdate:
            raise NotImplementedError()
            
        else:
            pass

    def add_single_model(self, sender_rank, content_list):
        """deal with single model's parameters"""
        self.client_model_queue.put(copy.deepcopy(content_list))

    def update_model(self, serialized_params_list=None):
        """"update global model from client_model_queue"""
        while not self.client_model_queue.empty():
            content_list = self.client_model_queue.get()
            self.adapt_alpha(content_list[1])
            receive_serialized_parameters = content_list[0]
            latest_serialized_parameters = SerializationTool.serialize_model(
                self.model)
            new_serialized_parameters = torch.mul(1 - self.alpha, latest_serialized_parameters) + \
                                        torch.mul(self.alpha, receive_serialized_parameters)
            SerializationTool.restore_model(self._model,
                                            new_serialized_parameters)

    def select_clients(self):
        """Return a list of client rank indices selected randomly"""
        id_list = [i + 1 for i in range(self.client_num_in_total)]
        selection = random.sample(id_list, self.client_num_per_round)
        return selection

    def adapt_alpha(self, receive_model_time):
        """update the alpha according to staleness"""
        self.alpha = torch.mul(self.alpha, 1)