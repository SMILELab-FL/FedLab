import logging
import random
import torch
import torch.distributed as dist

from torch.multiprocessing import Process
from fedlab_core.utils.messaging import MessageCode, send_message
from fedlab_core.utils.serialization import ravel_model_params

_LOGGER = logging.getLogger(__name__)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
_LOGGER.addHandler(handler)


class SSGDParameterServer():
    """
        ParameterServer
        同步参数服务器
    """
    def __init__(self, model, cuda=False, client_num=10):
        _LOGGER.info("Creating ParameterServer")
        self._model = model     # pytorch model
        self._buff = ravel_model_params(self._model)    # 序列化后的模型参数

        self.client_num = client_num  # 每轮参与者数量 定义buffer大小
        self.select_ratio = 1
        self.round_num = int(self.select_ratio*self.client_num)

        # buffer
        self.grad_buffer = [None for _ in range(self.client_num)]
        self.grad_buffer_cnt = 0

        # setup
        self.update_flag = False
        self.cuda = cuda
        
    def receive(self, sender, message_code, parameter) -> None:
        """
        上层调用
        开放接口
        """
        print("Processing message: {} from sender {}".format(message_code.name, sender))
        #_LOGGER.info("Processing message: {} from sender {}".format(message_code.name, sender))
        if message_code == MessageCode.ParameterUpdate:
            # 更新参数
            buffer_index = sender - 1
            if self.grad_buffer[buffer_index] is not None:
                return
            self.grad_buffer_cnt += 1
            self.grad_buffer[buffer_index] = parameter.clone()

            if self.grad_buffer_cnt == self.round_num:
                print("server model updated")
                self._buff[:] = torch.mean(
                    torch.stack(self.grad_buffer), dim=0)

                self.grad_buffer_cnt = 0
                self.grad_buffer = [None for _ in range(self.client_num)]

                self.update_flag = True

        elif message_code == MessageCode.ParameterRequest:
            # 请求参数
            raise Exception("SSGD should not be pull!!")

        elif message_code == MessageCode.GradientUpdate:
            # 梯度更新
            print("client update its gradient")

        elif message_code == MessageCode.Exit:
            exit(0)

        else:
            raise Exception("Undefined message type!")

    def is_updated(self) -> bool:
        return self.update_flag

    def select_clients(self):
        """
        上层调用
        """
        # 随机选取
        id_list = [i+1 for i in range(self.client_num)]
        select = random.sample(id_list, self.round_num)
        return select

    def get_buff(self):
        """
        上层调用
        """
        return self._buff

    def get_model(self):
        """
        上层调用
        """
        return self._model


class ASGDParameterServer():
    """ParameterServer
        异步参数服务器
    """

    def __init__(self, model):
        _LOGGER.info("Creating ParameterServer")
        print("Creating ParameterServer")
        self._model = model
        self.exit_flag = 0

    def receive(self, sender, message_code, parameter):
        _LOGGER.info("Processing message: {} from sender {}".format(
            message_code.name, sender))
        # print("Processing message: {} from sender {}".format(message_code.name, sender))

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
