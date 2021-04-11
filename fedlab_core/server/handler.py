import random
import torch
import torch.distributed as dist

from fedlab_core.utils.messaging import MessageCode, send_message
from fedlab_core.utils.serialization import ravel_model_params, unravel_model_params
from fedlab_core.utils.logger import logger


class ParameterServerHandler(object):
    """An abstract class representing handler for parameter server.

    Please make sure that you self-defined server handler class subclasses this class

    Example:
        read sourcecode of :class:`SyncSGDParameterServerHandler` below
    """

    def __init__(self, model, cuda=False) -> None:
        if cuda:
            self._model = model.cuda()
        else:
            self._model = model.cpu()
        self._buffer = ravel_model_params(self._model, cuda)
        self.cuda = cuda

    def on_receive(self):
        """Override this function to define what the server to do when receiving message from client"""
        raise NotImplementedError()

    def add_model(self, sender, payload):
        """Override this function to deal with incomming model

            Args:
                sender (int): the index of sender
                payload (): serialized model parameters
        """
        raise NotImplementedError()

    def update_model(self, model_list):
        """Override this function to update global model

        Args:
            model_list (list): a list of model parameters serialized by :func:`ravel_model_params`
        """
        raise NotImplementedError()

    @property
    def buffer(self):
        return self._buffer

    @property
    def model(self):
        return self._model

    @buffer.setter
    def buffer(self, buffer):
        """Update server model and buffer using serialized parameters"""
        unravel_model_params(self._model, buffer)
        self._buffer[:] = buffer[:]

    """
    @model.setter
    def model(self, model):
        #Update server model and buffer using serialized parameters
        # TODO: untested
        self._model[:] = model[:]
        self._buffer = ravel_model_params(self._model, self.cuda)
    """


class SyncParameterServerHandler(ParameterServerHandler):
    """Synchronous Parameter Server Handler

    Backend of synchronous parameter server: this class is responsible for backend computing.

    Synchronous parameter server will wait for every client to finish local training process before the
    next FL round.

    Args:
        model (torch.nn.Module): Model used in this federation
        client_num_in_total (int): Total number of clients in this federation
        cuda (bool): Use GPUs or not
        select_ratio (float): ``select_ratio * client_num`` is the number of clients to join every FL round
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

        self._LOGGER = logger(logger_path, logger_name)

        # client buffer
        self.client_buffer_cache = {}
        self.cache_cnt = 0

        # setup
        self.train_flag = False

    def on_receive(self, sender, message_code, payload) -> None:
        """Define what parameter server does when receiving a single client's message

        Args:
            sender (int): Index of client in distributed
            message_code (MessageCode): Agreements code defined in :class:`MessageCode` class
            payload (torch.Tensor): Serialized model parameters
        """
        self._LOGGER.info("Processing message: {} from sender {}".format(
            message_code.name, sender))

        if message_code == MessageCode.ParameterUpdate:
            # update model parameters
            self.add_model(sender, payload)
            # update server model when client_buffer_cache is full
            if self.cache_cnt == self.client_num_per_round:
                self.update_model(list(self.client_buffer_cache.values()))
        else:
            raise Exception("Undefined message type!")

    def select_clients(self):
        """Return a list of client rank indices selected randomly"""
        id_list = [i + 1 for i in range(self.client_num_in_total)]
        select = random.sample(id_list, self.client_num_per_round)
        return select

    def add_model(self, sender, payload):
        """deal with the model parameters"""
        buffer_index = sender - 1
        if self.client_buffer_cache.get(buffer_index) is not None:
            self._LOGGER.info("parameters from {} has existed".format(sender))
            return

        self.cache_cnt += 1
        self.client_buffer_cache[buffer_index] = payload.clone()

    def update_model(self, model_list):
        """update global model"""
        self._buffer[:] = torch.mean(
            torch.stack(model_list), dim=0)
        unravel_model_params(
            self._model, self._buffer)

        self.cache_cnt = 0
        self.client_buffer_cache = {}
        self.train_flag = False

    def train(self):
        self.train_flag = True


class AsyncParameterServerHandler(ParameterServerHandler):
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
        params = model_list[0]
        self._buffer[:] = (1 - self.alpha) * \
            self._buffer[:] + self.alpha * params
        unravel_model_params(self._model, self._buffer)  # load

    def on_receive(self, sender, message_code, parameter):

        if message_code == MessageCode.ParameterUpdate:
            self.update_model([parameter])

        elif message_code == MessageCode.ParameterRequest:
            send_message(MessageCode.ParameterUpdate, self._model, dst=sender)

        elif message_code == MessageCode.GradientUpdate:
            raise NotImplementedError()

        elif message_code == MessageCode.Exit:
            pass

        else:
            pass
