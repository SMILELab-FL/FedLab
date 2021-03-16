import os
import threading

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from fedlab_core.utils.messaging import MessageCode, recv_message, send_message


class ServerTop(Process):
    """
    Synchronise conmmunicate Class
       This is the top class in our framework which is mainly responsible for network communication of SERVER!.
       Synchronize with clients following agreements defined in run().
    """

    def __init__(self, ParameterServerHandler, server_addr, dist_backend="gloo", args=None):
        """constructor

        Args:
            ParameterServerHandler: a class derived from ParameterServerHandler
            args: params

        Returns:
            None
        Raises:
            None
        """
        self._params_server = ParameterServerHandler

        self.server_addr = server_addr
        self.dist_backend = dist_backend
        self.buff = torch.zeros(
            self._params_server.get_buff().numel() + 2).cpu()  # 通信信息缓存 模型参数+2
        self.args = args

    def run(self):
        """process function"""
        print("TPS|Waiting for the connection with clients!")
        # ip 127.0.0.1 port 3001
        dist.init_process_group(backend=self.dist_backend, init_method='tcp://{}:{}'
                                .format(self.server_addr[0], self.server_addr[1]), 
                                rank=0, world_size=self._params_server.client_num+1)
        print("TPS|Connect to client successfully!")

        #act_clients = threading.Thread(target=self.activate)
        #wait_info = threading.Thread(target=self.receive)
        self.running = True
        while self.running:
            print("TPS|Polling for message...")
            self.activate_clients()
            self.listen2clients()
            """
            # 开启选取参与者线程
            act_clients.start()
            # 开启接收回信线程
            wait_info.start()

            act_clients.join()
            wait_info.join()
            """

    def activate_clients(self):
        """activate some of clients to join this FL round"""
        print("activating...")
        usr_list = self._params_server.select_clients()
        payload = self._params_server.get_buff()
        for index in usr_list:
            send_message(MessageCode.ParameterUpdate, payload, dst=index)

    def listen2clients(self):
        """listen messages from clients"""
        self._params_server.start_round()
        while(True):
            recv_message(self.buff)
            sender = int(self.buff[0].item())
            message_code = MessageCode(self.buff[1].item())
            parameter = self.buff[2:]

            # handler 交给下层处理
            self._params_server.receive(sender, message_code, parameter)

            if self._params_server.is_updated():
                break
