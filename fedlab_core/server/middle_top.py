import logging
import os

import torch.distributed as dist
from torch.multiprocessing import Process, Lock, Value

from fedlab_core.server import SSGDParameterServer
from fedlab_core.server import PramsServer
from fedlab_core.utils.messaging import MessageCode, send_message, recv_message

_LOGGER = logging.getLogger(__name__)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
_LOGGER.addHandler(handler)


class ConnectTop(Process):
    def __init__(self, locks, model, args=None):
        self._lock = locks
        self.args = args
        self._n = 0
        super().__init__(model)

    def start(self):
        super().start()

    def run(self):
        print("SPS|Waiting for the connection with the top server!")
        dist.init_process_group(backend="gloo",
                                init_method='tcp://{}:{}'
                                .format(os.getenv("GLOBAL_SERVER_IP"), os.getenv("GLOBAL_SERVER_PORT")),
                                rank=int(os.getenv("GLOBAL_RANK")), world_size=int(os.getenv("GLOBAL_WORLD_SIZE")))
        print("SPS|Connect to the top server success!")
        while True:
            self._n += 1
            # 获取发送同步锁
            self._lock[0].acquire()

            if self.args.v.value:
                self.exit()

            if self.args.n_push > 0 and self._n % self.args.n_push == 0:
                # 发送给top
                send_message(MessageCode.GradientUpdate, self.model, dst=0)
                # 从top拿到更新`1
                self.model[:] = recv_message(self.buff)
            self._lock[1].release()

    def exit(self):
        send_message(MessageCode.Exit, self.model, dst=0)
        self._lock[1].release()
        print("SPS|For top exit!")
        exit(0)


# 类不均衡，标签错等。
# 数据好少、数据多但是差。
# 多只能提，采取行动的奖励如何分配。
class ConnectWorker(Process):
    def __init__(self, locks, model, args=None):
        self._lock = locks
        self._lock[0].acquire()
        self._lock[1].acquire()
        self._prams_server = None
        self.args = args
        super().__init__(model)

    def run(self):
        print("SPS|Waiting for the connection with local workers!")
        dist.init_process_group(backend="gloo",
                                init_method='tcp://{}:{}'
                                .format(os.getenv("LOCAL_SERVER_IP"), os.getenv("LOCAL_SERVER_PORT")),
                                rank=0, world_size=int(os.getenv("LOCAL_WORLD_SIZE")))
        print("SPS|Connect to local workers success!")
        self._prams_server = SSGDParameterServer(self.model)
        super().run()

    def receive(self, sender, message_code, parameter):
        # 更新模型参数
        # self.model[:] = parameter
        self._prams_server.receive(sender, message_code, parameter)
        print("1={}".format(parameter))
        # 通知后台和top进行同步
        if self._prams_server.can_sync():
            self._lock[0].release()
            # 获取同步结束得消息
            self._lock[1].acquire()
            # 同步后，发送参数
            self._prams_server.send()
            print("2={}".format(self.model))

        if self._prams_server.can_exit():
            self.args.can_exit = True
            self.args.v.value = True
            self._lock[0].release()
            print("SPS|For worker exit!")
            exit(0)
        # dist.send(self.model, dst=1)


class SecondServer(Process):
    def __init__(self, net, args):
        super().__init__()
        net.share_memory_()
        self._net = net
        self.args = args

    def run(self):
        locks = []
        [locks.append(Lock()) for _ in range(2)]
        self.args.v = Value('i', False)
        p1 = ConnectTop(locks, self._net, self.args, )
        p2 = ConnectWorker(locks, self._net, self.args)
        p1.start()
        p2.start()
        p1.join()
        p2.join()
