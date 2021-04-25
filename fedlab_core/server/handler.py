import os
import random
import torch

from fedlab_core.utils.logger import logger
from fedlab_core.utils.message_code import MessageCode
from fedlab_core.utils.serialization import SerializationTool


class ServerBackendHandler(object):
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

    def on_receive(self):
        """Override this function to define what the server to do when receiving message from client"""
        raise NotImplementedError()

    def add_single_model(self, sender, serialized_params):
        """Override this function to deal with incoming model

        Args:
            sender (int): rank of sender in distributed
            serialized_params (torch.Tensor): serialized model parameters
        """
        raise NotImplementedError()

    def update_model(self, serialized_params_list):
        """Override this function to update global model

        Args:
            serialized_params_list (list[torch.Tensor]): a list of serialized model parameters collected from different
        clients
        """
        raise NotImplementedError()

    def select_clients(self) -> list:
        raise NotImplementedError()

    @property
    def model(self):
        return self._model


class SyncParameterServerHandler(ServerBackendHandler):
    """Synchronous Parameter Server Handler

    Backend of synchronous parameter server: this class is responsible for backend computing.

    Synchronous parameter server will wait for every client to finish local training process before the
    next FL round.

    Args:
        model (torch.nn.Module): Model used in this federation
        client_num_in_total (int): Total number of clients in this federation
        cuda (bool): Use GPUs or not. Default: ``False``
        select_ratio (float): ``select_ratio * client_num`` is the number of clients to join every FL round. Default:
    ``1.0``
        logger_file (str, optional): Path to the log file for client handler. Default: ``"server_handler.txt"``
        logger_name (str, optional): Class name to initialize logger for client handler. Default: ``"server handler"``
    """

    def __init__(self, model, client_num_in_total, cuda=False, select_ratio=1.0, logger_path="server_handler.txt",
                 logger_name="server handler"):

        if select_ratio < 0.0 or select_ratio > 1.0:
            raise ValueError("Invalid select ratio: {}".format(select_ratio))

        if client_num_in_total < 1:
            raise ValueError(
                "Invalid total client number: {}".format(client_num_in_total))

        super(SyncParameterServerHandler, self).__init__(model, cuda)

        self.client_num_in_total = client_num_in_total
        self.select_ratio = select_ratio
        self.client_num_per_round = max(
            1, int(self.select_ratio * self.client_num_in_total))

        self._LOGGER = logger(os.path.join("log", logger_path), logger_name)

        # client buffer
        self.client_buffer_cache = {}
        self.cache_cnt = 0

        # setup
        self.train_flag = False

    def on_receive(self, sender, message_code, serialized_params) -> None:
        """Define what parameter server does when receiving a single client's message

        Args:
            sender (int): Rank of client process sending this local model
            message_code (MessageCode): Agreements code defined in :class:`MessageCode`
            serialized_params (torch.Tensor): Serialized local model parameters from client
        """

        self._LOGGER.info("Processing message: {} from rank {}".format(
            message_code.name, int(sender)))

        if message_code == MessageCode.ParameterUpdate:
            # update model parameters
            self.add_single_model(sender, serialized_params)
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

    def add_single_model(self, sender, serialized_params):
        """deal with single model's parameters"""
        buffer_index = sender - 1
        if self.client_buffer_cache.get(buffer_index) is not None:
            self._LOGGER.info("parameters from {} has existed".format(sender))
            return

        self.cache_cnt += 1
        self.client_buffer_cache[buffer_index] = serialized_params.clone()

    def update_model(self, serialized_params_list):
        """update global model"""
        serialized_parameters = torch.mean(torch.stack(serialized_params_list), dim=0)
        SerializationTool.restore_model(self._model, serialized_parameters)

        # reset
        self.cache_cnt = 0
        self.client_buffer_cache = {}
        self.train_flag = False

    def train(self):
        self.train_flag = True


class AsyncParameterServerHandler(ServerBackendHandler):
    """Asynchronous ParameterServer Handler

    Update global model immediately after receiving a ParameterUpdate message
    paper: https://arxiv.org/abs/1903.03934

    Args:
        model (torch.nn.Module): Global model in server
        cuda (bool): Use GPUs or not
    """

    # TODO: unfinished
    def __init__(self, model, cuda):
        super(AsyncParameterServerHandler, self).__init__(model, cuda)
        self.client_buffer_cache = []
        self.alpha = 0.5
        self.decay = 0.9

    def update_model(self, model_list):
        raise NotImplementedError()

    def on_receive(self, sender, message_code, parameter):

        if message_code == MessageCode.ParameterUpdate:
            self.update_model([parameter])

        elif message_code == MessageCode.ParameterRequest:
            # send_message(MessageCode.ParameterUpdate, self._model, dst=sender)
            pass

        elif message_code == MessageCode.GradientUpdate:
            raise NotImplementedError()

        elif message_code == MessageCode.Exit:
            pass

        else:
            pass
