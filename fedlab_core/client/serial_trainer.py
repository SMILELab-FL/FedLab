from copy import deepcopy
import threading
import logging
from time import time

import torch

from fedlab_utils.logger import logger
from fedlab_utils.serialization import SerializationTool
from fedlab_utils.dataset.sampler import SubsetSampler
from fedlab_core.client.trainer import ClientTrainer


class SerialTrainer(ClientTrainer):
    """Train multiple clients with a single process or multiple threads.

    Args:
        model (nn.Module): Model used in this federation.
        aggregator (fedlab_utils.aggregator): function to deal with a list of parameters.
        dataset (nn.utils.dataset): local dataset for this group of clients.
        data_slices (list): subset of indices of dataset.
        logger (:class:`fedlab_utils.logger`, optional): an util class to print log info to specific file and cmd line. If None, only cmd line. 
    
    Notes:
        len(data_slices) == client_num, which means that every sub-indices of dataset represents a client's local dataset.
        
    """
    def __init__(self,
                 model: torch.nn.Module,
                 dataset,
                 data_slices,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 aggregator,
                 logger: logger = None,
                 cuda: bool = True) -> None:

        super(SerialTrainer, self).__init__(model=model, cuda=cuda)

        self.dataset = dataset
        self.data_slices = data_slices  #[0,sim_client_num)
        self.client_num = len(data_slices)

        self.criterion = criterion
        self.optimizer = optimizer
        self.aggregator = aggregator

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def _get_dataloader(self, client_id, batch_size):
        """Return a dataloader used in :meth:`train`

        Args:
            client_id (int): client id to generate this dataloader
            batch_size (int): batch size
        
        Returns:
            Dataloader for specific client sub-dataset
        """
        trainloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[client_id - 1],
                                  shuffle=True),
            batch_size=batch_size)
        return trainloader

    def _train_alone(self, model_parameters, epochs, batch_size, lr, cuda):
        # lock
        model = deepcopy(self.model)
        self.optimizer.load_state_dict(model.parameters())
        # unlock
        SerializationTool.deserialize_model(model, model_parameters)
        data_loader = self._get_dataloader(id, batch_size)

        # classic train pipeline
        model.train()
        for epoch in range(epochs):
            loss_sum = 0.0
            time_begin = time()
            for _, (data, target) in enumerate(data_loader):
                if cuda:
                    data = data.cuda()
                    target = target.cuda()

                output = self.model(data)

                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.detach().item()

            print("Client[{}] Traning. Epoch {}/{}, Loss {:.4f}, Time {:.2f}s".
                  format(id, epoch + 1, epochs, loss_sum,
                         time() - time_begin))

    def train(self,
              model_parameters,
              epochs,
              lr,
              batch_size,
              id_list,
              cuda,
              multi_threading=False):
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
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        param_list = []
        for id in id_list:
            self._LOGGER.info(
                "starting training process of client [{}]".format(id))
            SerializationTool.deserialize_model(self._model, model_parameters)
            data_loader = self._get_dataloader(id, batch_size)

            # classic train pipeline
            self._model.train()
            for epoch in range(epochs):
                loss_sum = 0.0
                time_begin = time()
                for step, (data, target) in enumerate(data_loader):
                    if cuda:
                        data = data.cuda()
                        target = target.cuda()

                    output = self.model(data)

                    loss = criterion(output, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_sum += loss.detach().item()

                self._LOGGER.info(
                    "Client[{}] Traning. Epoch {}/{}, Loss {:.4f}, Time {:.2f}s"
                    .format(id, epoch + 1, epochs, loss_sum,
                            time() - time_begin))
            param_list.append(SerializationTool.serialize_model(self.model))

        # aggregate model parameters
        return self.aggregator(param_list)
