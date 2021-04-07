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

    def receive(self):
        """Override this function to define what the server to do when receiving message from client"""
        raise NotImplementedError()

    @property
    def buffer(self):
        return self._buffer

    @property
    def model(self):
        return self._model


class SyncParameterServerHandler(ParameterServerHandler):
    """Synchronize Parameter Server Handler
        Backend of parameter server: this class is responsible for backend computing
        Synchronize ps(parameter server) will wait for every client finishing their local training process before the next FL round

    Args:
        model: torch.nn.Module
        client_num: the number of client in this federation
        cuda: use GPUs or not
        select_ratio: select_ratio*client_num is the number of clients to join every FL round

    Raises:
        None
    """

    def __init__(self, model, client_num, cuda=False, select_ratio=1.0, logger_path="server_handler.txt",
                 logger_name="server handler"):
        super(SyncParameterServerHandler, self).__init__(model, cuda)

        self.client_num = client_num  # 每轮参与者数量 定义buffer大小
        self.select_ratio = select_ratio
        self.round_num = int(self.select_ratio * self.client_num)

        self._LOGGER = logger(logger_path, logger_name)

        # client buffer
        self.client_buffer_cache = [None for _ in range(self.client_num)]
        self.buffer_cnt = 0

        # setup
        self.update_flag = False

    def receive(self, sender, message_code, payload) -> None:
        """Define what parameter server does when receiving a single client's message

        Args:
            sender (int): Index of client in distributed
            message_code: agreements code defined in :class:`MessageCode` class
            payload (torch.Tensor): Serialized model parameters, obtained from :func:`ravel_model_params`

        Returns:
            None

        Raises:
            None

        """
        self._LOGGER.info("Processing message: {} from sender {}".format(
            message_code.name, sender))

        if message_code == MessageCode.ParameterUpdate:
            # update model parameters
            buffer_index = sender - 1
            if self.client_buffer_cache[buffer_index] is not None:
                self._LOGGER.info(
                    "parameters from {} has exsited".format(sender))
                return

            self.buffer_cnt += 1
            self.client_buffer_cache[buffer_index] = payload.clone()

            if self.buffer_cnt == self.round_num:
                """ if `client_buffer_cache` is full, then update server model"""
                self._buffer[:] = torch.mean(
                    torch.stack(self.client_buffer_cache), dim=0)

                unravel_model_params(self._model, self._buffer)  # 通过buffer更新全局模型

                self.buffer_cnt = 0
                self.client_buffer_cache = [
                    None for _ in range(self.client_num)]
                self.update_flag = True
        else:
            raise Exception("Undefined message type!")

    def select_clients(self):
        """Return a list of client rank indices"""
        id_list = [i + 1 for i in range(self.client_num)]
        select = random.sample(id_list, self.round_num)
        return select

    def is_updated(self) -> bool:
        return self.update_flag

    def start_round(self):
        self.update_flag = False


# TODO: finish Async class
class AsyncParameterServer():
    """ParameterServer
        TODO: this class is not implemented yet
    """

    def __init__(self, model):
        # _LOGGER.info("Creating ParameterServer")
        raise NotImplementedError()
        print("Creating ParameterServer")
        self._model = model
        self.exit_flag = 0

    def receive(self, sender, message_code, parameter):

        if message_code == MessageCode.ParameterUpdate:
            # be sure to clone here
            self._model[:] = parameter

        elif message_code == MessageCode.ParameterRequest:
            send_message(MessageCode.ParameterUpdate, self._model, dst=sender)

        elif message_code == MessageCode.GradientUpdate:
            self._model.add_(parameter)

        elif message_code == MessageCode.Exit:
            self.exit_flag += 1
            if self.exit_flag == dist.get_world_size():
                exit(0)

    def can_sync(self) -> bool:
        return False

    def send(self, sender) -> None:
        raise Exception("ASGD Server's sending option in recive")
        # send_message(MessageCode.ParameterUpdate, self._model, dst=sender)
