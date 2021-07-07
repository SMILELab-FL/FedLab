import time
from abc import ABC, abstractmethod
import logging

import torch
from torch import nn

from fedlab_utils.logger import logger
from fedlab_utils.serialization import SerializationTool


class ClientTrainer(ABC):
    """An abstract class representing handler for a client backend.

    In our framework, we define the backend of client handler show manage its local model.
    It should have a function to update its model called :meth:`train`.

    If you use our framework to define the activities of client, please make sure that your self-defined class
    should subclass it. All subclasses should overwrite :meth:`train`.

    Args:
        model (torch.nn.Module): Model used in this federation
        cuda (bool): Use GPUs or not
    """
    def __init__(self, model: nn.Module, cuda: bool):
        self.cuda = cuda
        if self.cuda:
            self._model = model.cuda()
        else:
            self._model = model.cpu()

    @abstractmethod
    def train(self):
        """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
        raise NotImplementedError()

    @property
    def model(self):
        """attribute"""
        return self._model


class ClientSGDTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): model used in federation
        data_loader (torch.Dataloader): :class:`DataLoader` for this client
        epoch (int): local epoch
        optimizer (torch.optim.Optimizer, optional): optimizer for this client's model. If set to ``None``, will use :func:`torch.optim.SGD` with :attr:`lr` of 0.1 and :attr:`momentum` of 0.9 as default.
        criterion (optional): loss function used in local training process. If set to ``None``, will use:func:`nn.CrossEntropyLoss` as default.
        cuda (bool, optional): use GPUs or not. Default: ``True``
        logger (optional): `fedlab_utils.logger`, 

    """
    def __init__(self,
                 model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 epoch: int,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 cuda: bool = True,
                 logger: logger = None):
        super(ClientSGDTrainer, self).__init__(model, cuda)

        self._data_loader = data_loader

        self.epoch = epoch
        self.optimizer = optimizer
        self.criterion = criterion

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def train(self, model_parameters: torch.Tensor, epoch: int = None) -> None:
        """
        Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): serialized model paremeters
            epochs (int): number of epoch for current local training
        """
        self._LOGGER.info("starting local train process")
        SerializationTool.deserialize_model(self._model,
                                            model_parameters)  # load paramters

        if epoch is None:
            epochs = self.epoch
        else:
            epochs = epoch

        for epoch in range(epochs):
            start_time = time.time()
            self._model.train()
            loss_sum = 0.0
            for inputs, labels in self._data_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.detach().item()
            end_time = time.time()

            self._LOGGER.info(
                "Epoch {}/{}, Loss: {:.4f}, Time cost: {:.2f}s".format(
                    epoch + 1, epochs, loss_sum, end_time - start_time))
