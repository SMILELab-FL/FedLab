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
from fedlab_utils.dataset.sampler import AssignSampler

class SerialHandler(object):
    """Train multiple clients with a single process.

    Args:
        local_model (nn.Module): Model used in this federation.
        aggregator (fedlab_utils.aggregator): function to deal with a list of parameters.
        dataset (nn.utils.dataset): local dataset for this group of clients.
        sim_client_num (int): the number of client this class should maintain.
        logger (:class:`fedlab_utils.logger`, optional): an util class to print log info to specific file and cmd line. If None, only cmd line. 
    
    Attributes:
        model:
        aggregator:
        sim_client_num:
        optimizer:
        criterion:
        trainset:
        data_slices:
        _LOGGER:
    """
    def __init__(self, local_model, aggregator, dataset, sim_client_num, client_data_indices, lr=0.1, logger=None) -> None:
        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
        
        self.aggregator = aggregator
        self.model = local_model
        self.sim_client_num = sim_client_num

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.trainset = dataset
        self.data_slices = client_data_indices #[0,sim_client_num)

        self._LOGGER = logging if logger is None else logger

    def _get_dataloader(self, client_id, batch_size):
        """Return a dataloader used in :meth:`train`

        Args:
            client_id (int): client id to generate this dataloader
            batch_size (int): batch size
        
        Returns:
            Dataloader for specific client sub-dataset
        """
        trainloader = torch.utils.data.DataLoader(self.trainset, sampler = AssignSampler(indices=self.data_slices[client_id-1], shuffle=True), batch_size=batch_size)
        return trainloader

    def train(self, epochs, model_parameters, batch_size, id_list, cuda):
        """Train local model with different dataset according to id in id_list.

        Args:
            epochs (int): number of epoch for local training.
            model_parameters (torch.Tensor): serialized model paremeters.
            batch_size (int):
            id_list (list): client id in this train 
            cuda (bool): use GPUs or not.

        Returns:
            Merged serialized params
        """
        param_list = []
        for id in id_list:
            self._LOGGER.info("starting training process of client [{}]".format(id))
            SerializationTool.deserialize_model(self.model, model_parameters)
            data_loader = self._get_dataloader(id, batch_size)

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
        """Train multiple clients with multiple threads."""
        pass

        