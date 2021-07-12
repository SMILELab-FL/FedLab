from copy import deepcopy
import threading
import logging
from time import time

import torch

from fedlab_utils.logger import logger
from fedlab_utils.serialization import SerializationTool
from fedlab_utils.dataset.sampler import SubsetSampler
from fedlab_core.client.trainer import ClientTrainer


class ReturnThread(threading.Thread):
    def __init__(self, target, args):
        super(ReturnThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


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
                 aggregator,
                 logger: logger = None,
                 cuda: bool = True) -> None:

        super(SerialTrainer, self).__init__(model=model, cuda=cuda)

        self.dataset = dataset
        self.data_slices = data_slices  #[0,sim_client_num)
        self.client_num = len(data_slices)
        self.aggregator = aggregator

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def _get_dataloader(self, id, batch_size):
        """Return a dataloader used in :meth:`train`

        Args:
            client_id (int): client id to generate this dataloader
            batch_size (int): batch size
        
        Returns:
            Dataloader for specific client sub-dataset
        """
        trainloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[id - 1],
                                  shuffle=True),
            batch_size=batch_size)
        return trainloader

    def _train_alone(self, id, model, epochs, data_loader, optimizer,
                     criterion, cuda):
        """single round of training

        Args:
            id (int): client id of this round.
            model (nn.Module): model to be trained.
            epochs (int): the local epoch of training.
            data_loader (torch.utils.data.DataLoader): dataloader for data iteration.
            optimizer (torch.Optimizer): Optimizer associated with model.
            critereion (torch.nn.Loss): loss function.
            cuda (bool): use GPUs or not.
        """
        model.train()
        for epoch in range(epochs):
            loss_sum = 0.0
            time_begin = time()
            for _, (data, target) in enumerate(data_loader):
                if cuda:
                    data = data.cuda()
                    target = target.cuda()

                output = self.model(data)

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.detach().item()

            print("Client[{}] Traning. Epoch {}/{}, Loss {:.4f}, Time {:.2f}s".
                  format(id, epoch + 1, epochs, loss_sum,
                         time() - time_begin))
        return SerializationTool.serialize_model(model)

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

        #TODO: something wrong with multi_threading.
                异步训练失败？
        """
        param_list = []

        if multi_threading is True:
            threads = []

        for id in id_list:
            self._LOGGER.info(
                "starting training process of client [{}]".format(id))

            SerializationTool.deserialize_model(self._model, model_parameters)
            criterion = torch.nn.CrossEntropyLoss()
            data_loader = self._get_dataloader(client_id=id,
                                               batch_size=batch_size)

            if multi_threading is False:

                optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
                self._train_alone(id,
                                  self._model,
                                  epochs=epochs,
                                  data_loader=data_loader,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  cuda=cuda)
                param_list.append(SerializationTool.serialize_model(
                    self.model))

            else:

                model = deepcopy(self._model)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                args = (id, model, epochs, data_loader, optimizer, criterion,
                        cuda)
                t = ReturnThread(target=self._train_alone, args=args)
                t.start()
                threads.append(t)

        if multi_threading is True:
            for t in threads:
                t.join()
            for t in threads:
                param_list.append(t.get_result())

        # aggregate model parameters
        return self.aggregator(param_list)
