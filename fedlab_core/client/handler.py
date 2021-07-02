import time
from abc import ABC, abstractmethod
import logging

import torch
from torch import nn

from  fedlab_utils.serialization import SerializationTool


class ClientBackendHandler(ABC):
    """An abstract class representing handler for a client backend.

    In our framework, we define the backend of client handler show manage its local model.
    It should have a function to update its model called :meth:`train`.

    If you use our framework to define the activities of client, please make sure that your self-defined class
    should subclass it. All subclasses should overwrite :meth:`train`.

    Args:
        model (torch.nn.Module): Model used in this federation
        cuda (bool): Use GPUs or not

    Attributes:
        _model (torch.nn.Module): 
        cuda (bool):
    """
    def __init__(self, model, cuda):
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
        """  """
        return self._model


class ClientSGDHandler(ClientBackendHandler):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): model used in federation
        data_loader (torch.Dataloader): :class:`DataLoader` for this client
        optimizer (torch.optim.Optimizer, optional): optimizer for this client's model. If set to ``None``, will use
        :func:`torch.optim.SGD` with :attr:`lr` of 0.1 and :attr:`momentum` of 0.9 as default.
        criterion (optional): loss function used in local training process. If set to ``None``, will use
        :func:`nn.CrossEntropyLoss` as default.
        cuda (bool, optional): use GPUs or not. Default: ``True``
        logger (optional): `fedlab_utils.logger`, 
    
    Attributes:
        _model (torch.nn.Module): 
        cuda (bool):
        _data_loader:
        _LOGGER:
        epoch:
        optimizer:
        criterion:
    """
    def __init__(self, model, data_loader, local_epoch, optimizer=None, criterion=None, cuda=True, logger=None):
        super(ClientSGDHandler, self).__init__(model, cuda)

        self._data_loader = data_loader
        self.epoch = local_epoch

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger
            
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self._model.parameters(), lr=0.1, momentum=0.9)
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def train(self, model_parameters, epochs=None):
        """
        Client trains its local model on local dataset.

        Args:
            epochs (int): number of epoch for local training
            model_parameters (torch.Tensor): serialized model paremeters
        """
        self._LOGGER.info("starting local train process")
        SerializationTool.deserialize_model(self._model, model_parameters) # load paramters

        if epochs is None:
            local_epoch = self.epoch
        else:
            local_epoch = epochs
    
        for epoch in range(local_epoch):
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

            self._LOGGER.info("Epoch {}/{}, Loss: {:.4f}, Time cost: {:.2f}s".format(epoch + 1, local_epoch, loss_sum, end_time-start_time))

    

    
