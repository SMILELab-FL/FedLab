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

from  fedlab_utils.serialization import SerializationTool
from fedlab_utils.dataset.sampler import DistributedSampler
from fedlab_utils.dataset.sampler import AssignSampler

class SerialHandler(object):
    """
    在FedLab架构中作为单个client，逻辑上可以模拟任意数量的参与者模型的训练
    串行IO

    Args:
        local_model ():
        aggregator ():
        dataset ():
        sim_client_num ():
        logger ():
    """
    def __init__(self, local_model, aggregator, dataset, sim_client_num, client_data_indices, lr=0.1, logger=None) -> None:
        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
        
        self.aggregator = aggregator
        self.model = local_model
        self.sim_client_num = sim_client_num
        self.data_slices = client_data_indices

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.trainset = dataset
        self._LOGGER = logging if logger is None else logger

    def get_dataloader(self, client_id, batch_size):
        """
        Args:
            client_id ():
            batch_size ():
            sampler ():
        """
        trainloader = torch.utils.data.DataLoader(self.trainset, sampler = AssignSampler(indices=self.data_slices[client_id], shuffle=True), batch_size=batch_size)
        return trainloader

    def train(self, epochs, batch_size, idx_list, model_parameters, cuda):
        """
        """
        param_list = []
        for id in idx_list:
            self._LOGGER.info("starting training process of client [{}]".format(id))
            SerializationTool.deserialize_model(self.model, model_parameters)
            data_loader = self.get_dataloader(id, batch_size)

            # classic train pipeline
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
        
        # aggregate model parameters
        return self.aggregator(param_list)

    def multithreading_train(self):
        pass

        