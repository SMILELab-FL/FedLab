import os
import threading

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from FLTB_core.server import PramsServer
from FLTB_core.utils.messaging import MessageCode, recv_message, send_message


class ServerTop(Process):
    """
    仅负责网络通信的任务
    底层参数的处理交给self._params_server, 因此需要初始化底层算法逻辑
    """
    def __init__(self, ParameterServer, args):
        self._params_server = ParameterServer
        self.buff = torch.zeros(self._params_server.get_buff().numel() + 2).cpu()  # 通信信息缓存 模型参数+2

    def run(self):
        """
        TopServer进程函数
        """
        print("TPS|Waiting for the connection with clients!")
        dist.init_process_group(backend="gloo", init_method='tcp://{}:{}'
                                .format("127.0.0.1", 3001),
                                rank=0, world_size=self._params_server.client_num+1)
        print("TPS|Connect to client successfully!")

        act_clients = threading.Thread(target=self.activate)
        wait_info = threading.Thread(target=self.receive)
        self.running = True
        while self.running:
            print("TPS|Polling for message...")
            # 开启选取参与者线程
            act_clients.start()
            # 开启接收回信线程
            wait_info.start()

            act_clients.join()
            wait_info.join()


    def activate(self):
        """
        开放接口
        """
        usr_list = self._params_server.select_clients()
        payload = self._params_server.get_buff()
        for index in usr_list:
            send_message(MessageCode.ParameterUpdate, payload, dst=index)


    def receive(self, sender, message_code, parameter):
        """
        开放接口
        """
        while(True):
            recv_message(self.buff)
            sender = int(self.buff[0].item())
            message_code = MessageCode(self.buff[1].item())
            parameter = self.buff[2:]
            
            # handler 交给下层处理
            self._params_server.receive(sender, message_code, parameter)

            if self._params_server.is_updated():
                break
