from math import log
import os
from random import triangular
import time

import torch
from torch import nn
from fedlab_core.utils.serialization import ravel_model_params, unravel_model_params
from fedlab_core.utils.ex_recorder import ExRecorder
from fedlab_core.utils.logger import logger


class ClientBackendHandler(object):
    """An abstract class representing handler for a client backend.

    In our framework, we define the backend of client handler show manage its local model and buffer.
    It should have a function to update its model called :meth:`train` and a function called :meth:`evaluate`.

    If you use our framework to define the activities of client, please make sure that your self-defined class
    should subclass it. All subclasses should overwrite :meth:`train` and :meth:`evaluate`.
    """

    def __init__(self, model, cuda):
        self.cuda = cuda

        if self.cuda:
            self._model = model.cuda()
        else:
            self._model = model.cpu()

        self._buffer = ravel_model_params(model, self.cuda)

    @property
    def model(self):
        """Get :class:`torch.nn.Module`"""
        return self._model

    @property
    def buffer(self):
        """Get serialized parameters"""
        return self._buffer

    @buffer.setter
    def buffer(self, buffer):
        """Update local model and buffer using serialized parameters"""
        unravel_model_params(self._model, buffer)
        self._buffer[:] = buffer[:]

    @model.setter
    def model(self, model):
        """Update local model and buffer using serialized parameters"""
        # TODO: untested
        self._model[:] = model[:]
        self._buffer = ravel_model_params(self._model, self.cuda)

    def train(self):
        """Please override this method. This function should manipulate :attr:`self._model` and :attr:`self._buffer`"""
        raise NotImplementedError()

    def evaluate(self, test_loader):
        """Please override this method. Evaluate local model based on given test :class:`torch.DataLoader"""
        raise NotImplementedError()


class ClientSGDHandler(ClientBackendHandler):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module):
        data_loader (torch.Dataloader): :class:`DataLoader` for this client
        optimizer (torch.optim.Optimizer, optional): optimizer for this client's model. If set to ``None``, will use
        :func:`torch.optim.SGD` with :attr:`lr` of 0.1 and :attr:`momentum` of 0.9 as default.
        criterion (optional): loss function used in local training process. If set to ``None``, will use
        :func:`nn.CrossEntropyLoss` as default.
        cuda (bool, optional): use GPUs or not. Default: ``True``
        LOGGER: utils class to output debug information to file and command line

    Raises:
        None
    """
    # TODO: fix the logger parameter

    def __init__(self, model, data_loader, optimizer=None, criterion=None, cuda=True, logger_file="handler.txt",
                 logger_name="handler"):
        super(ClientSGDHandler, self).__init__(model, cuda)

        self._data_loader = data_loader

        self._LOGGER = logger(logger_file, logger_name)

        if optimizer is None:
            self.optimizer = torch.optim.SGD(
                self._model.parameters(), lr=0.1, momentum=0.9)
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def train(self, epochs):
        """
        Client trains its local model on local dataset.

        Args:
            epochs (int): number of epoch for local training
        """

        self._LOGGER.info("starting local train process")
        for epoch in range(epochs):
            self._model.train()
            for inputs, labels in self._data_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                self.optimizer.zero_grad()

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
            log_str = "Epoch {}/{}".format(epoch + 1, epochs)  # TODO: loss record?
            self._LOGGER.info(log_str)

        self._buffer = ravel_model_params(self._model, cuda=True)

    def evaluate(self, test_loader, cuda):
        # TODO: Evaluate local model based on given test `torch.DataLoader`

        self._model.eval()
        loss_sum = 0.0
        for input, label in test_loader:
            if cuda:
                input = input.cuda()
                label = label.cuda()

            with torch.no_grad():
                out = self._model(input)
                loss = self.criterion(out, label)

            loss_sum += loss.item()
            # TODO: finish this
            # TODO: why not use `with torch.no_grad():` to wrap all other loops?
