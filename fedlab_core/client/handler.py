import time

import torch
from torch import nn
from fedlab_core.utils.logger import logger
from fedlab_core.message_processor import SerializationTool


class ClientBackendHandler(object):
    """An abstract class representing handler for a client backend.

    In our framework, we define the backend of client handler show manage its local model and buffer.
    It should have a function to update its model called :meth:`train` and a function called :meth:`evaluate`.

    If you use our framework to define the activities of client, please make sure that your self-defined class
    should subclass it. All subclasses should overwrite :meth:`train` and :meth:`evaluate`.

    Args:
        model (torch.nn.Module): Model used in this federation
        cuda (bool): Use GPUs or not

    """

    def __init__(self, model, cuda):
        self.cuda = cuda
        if self.cuda:
            self._model = model.cuda()
        else:
            self._model = model.cpu()

    def train(self):
        """Please override this method. This function should manipulate :attr:`self._model`"""
        raise NotImplementedError()

    def evaluate(self, test_loader):
        """Please override this method. Evaluate local model based on given test :class:`torch.DataLoader`"""
        raise NotImplementedError()

    def load_parameters(self, serialized_parameters):
        """Restore model from serialized model parameters"""
        SerializationTool.restore_model(self._model, serialized_parameters)

    @property
    def model(self):
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
        logger_file (str, optional): Path to the log file for client handler. Default: ``"log/handler.txt"``
        logger_name (str, optional): Class name to initialize logger for client handler. Default: ``"handler"``
    """

    def __init__(self, model, data_loader, optimizer=None, criterion=None, cuda=True, logger_file="log/handler.txt",
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
            loss_sum = 0.0
            for inputs, labels in self._data_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                self.optimizer.zero_grad()

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                loss_sum += loss.detach().item()

            # TODO: is it proper to use loss_sum here?? CrossEntropyLoss is averaged over each sample
            log_str = "Epoch {}/{}, Loss: {}, Time: {}".format(
                epoch + 1, epochs, loss_sum, time.time())
            self._LOGGER.info(log_str)

    def evaluate(self, test_loader, cuda):
        """
        Evaluate local model based on given test :class:`torch.DataLoader`

        Args:
            test_loader (torch.DataLoader): :class:`DataLoader` for evaluation
            cuda (bool): Use GPUs or not
        """
        self._model.eval()
        loss_sum = 0.0
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                if cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct += torch.sum(predicted.eq(labels)).item()
                total += len(labels)
                loss_sum += loss.item()

        accuracy = correct / total
        # TODO: is it proper to use loss_sum here?? CrossEntropyLoss is averaged over each sample
        log_str = "Evaluate, Loss {}, accuracy: {}".format(loss_sum,
                                                           accuracy)
        self._LOGGER.info(log_str)
