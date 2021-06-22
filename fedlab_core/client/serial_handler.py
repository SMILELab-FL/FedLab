# 单进程串行模拟多client后端
# 进程资源限制
# serial_handler仅share一个网络topology模块
# serial_handler对上层提供

#untested


import random

from abc import ABC, abstractmethod
from copy import deepcopy
import threading
import logging
from time import time

import torch

from fedlab_core.client.handler import ClientBackendHandler
from  fedlab_utils.serialization import SerializationTool
from fedlab_utils.dataset.sampler import DistributedSampler

class SerialHandler(ABC):
    """An abstract class for serial model training process
    
    It is expensive to simulate a single client per process, because of process exchange in OS.
    Therefore, we need a class to simulate multiple clients with a process resource.

    Subclass this class to create your own simulate algorithm.

    Args:
        model (torch.nn.Module): Model used in this federation
        cuda (bool): use GPUs or not
    """ 
    def __init__(self, local_model, aggregator) -> None:
        
        self.aggregator = aggregator
        self.model = local_model

    @abstractmethod
    def train(self, epochs, model_parameters, idx_list=None):
        raise NotImplementedError()
    
    @abstractmethod
    def _merge_models(self, idx_list):
        raise NotImplementedError()

    @property
    def model(self):
        return self.model
        

class SerialMultipleHandler(object):
    """an example of SerialHandler
        Every client in this Serial share the same shape of model. Each of them has different
        datasets (differences shows in the init of ClientHandler)
        
        This class should be a perfect replace of ClientBackendHandler. Therefore, given the same methods to upper class.

        Args：
            client_handler_list (list): list of objects of ClientBackendHandler's subclass

        弊端 优化进程切换的io， 显存占用不变
            可多线程计算，并发性不受影响
    """
    def __init__(self, client_handler_list, multi_thread=False) -> None:
        super(SerialHandler, self).__init__()
        
        self.clients = client_handler_list
        self.serial_number = len(client_handler_list)
        self.multi_thread = multi_thread

    def train(self, epochs, model_parameters, idx_list=None, select_ratio=None):

        if idx_list is None:
            idx_list = [i for i in range(self.serial_number)]
            if select_ratio is not None:
                select_ratio = min(1, max(0,select_ratio))
                idx_list = random.sample(idx_list, max(1, int(select_ratio*self.serial_number)) )
            else:
                raise ValueError("idx_list and select_ration can't be None at the same time!")

        if self.multi_thread is True:
            thread_pool = []
        
        for idx in idx_list:
            assert idx<self.serial_number
            if idx >= self.serial_number:
                raise ValueError("Invalid idx of client: %d >= %d"%(idx, self.serial_number))
            self.clients[idx].train(epochs, model_parameters)


        merged_parameters = self._merge_models(idx_list)
        SerializationTool.deserialize_model(self.model, merged_parameters)

    def _merge_models(self, idx_list):
        parameter_list = [SerializationTool.serialize_model(self.clients[idx].model) for idx in idx_list]
        merged_parameters = self.aggregator(parameter_list)
        return merged_parameters

    

class SerialSingleHandler(object):
    """
    多个模型share一个显存模型
    子client仅维护参数，而不是一个可推理的模型

    该类仅支持单进程序列化学习

    已测试
    """
    def __init__(self, local_model, aggregator, dataset, sim_client_num, logger=None) -> None:
        self.aggregator = aggregator
        self.model = local_model
        self.sim_client_num = sim_client_num
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.trainset = dataset
        self._LOGGER = logging if logger is None else logger

    def get_dataloader(self, client_id, batch_size):
        trainloader = torch.utils.data.DataLoader(self.trainset, sampler = DistributedSampler(self.trainset, rank=client_id, num_replicas=self.sim_client_num), batch_size=batch_size)
        return trainloader

    def train(self, epochs, batch_size, idx_list, model_parameters, cuda):
        param_list = []
        for id in idx_list:
            self._LOGGER.info("starting training process of client [{}]".format(id))
            SerializationTool.deserialize_model(self.model, model_parameters)
            data_loader = self.get_dataloader(id, batch_size)
            self.model.train()
            for epoch in range(epochs):
                loss_sum = 0.0
                time_begin = time()
                for step, (data, target) in enumerate(data_loader):
                    if cuda:
                        data = data.cuda()
                        target = target.cuda()
                    
                    output = self.model(data)

                    loss = self.criterion(output, target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_sum += loss.detach().item()
                    
                self._LOGGER.info("Client[{}] Traning. Epoch {}/{}, Loss {:.4f}, Time {:.2f}s".format(id, epoch+1,epochs, loss_sum, time()-time_begin))
            param_list.append(SerializationTool.serialize_model(self.model))
        
        return self.aggregator(param_list)


        