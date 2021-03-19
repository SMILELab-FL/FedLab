import os
from random import triangular
import time

import torch
from torch import nn
from fedlab_core.utils.serialization import ravel_model_params, unravel_model_params
from fedlab_core.utils.ex_recorder import ExRecorder


class ClientBackendHandler(object):
    """Abstract class"""
    def __init__(self, model, cuda):
        if cuda:
            self._model = model.cuda()
        else:
            self._model = model.cpu()

        self._buffer = ravel_model_params(model, cuda)

    @property
    def model(self):
        """Get `torch.nn.Module`"""
        return self._model

    @property
    def buffer(self):
        """Get serialized parameters"""
        return self._buffer

    @buffer.setter
    def buffer(self, buffer):
        """Update local model using serialized parameters"""
        unravel_model_params(self._model, buffer)
        self._buff[:] = buffer

    def train(self):
        # TODO: please override this function. This
        #  function should manipulate self._model and self._buffer
        raise NotImplementedError()

    def evaluate(self, test_loader):
        # TODO: please override this function. Evaluate local model
        #  based on given test torch.DataLoader
        raise NotImplementedError()


class ClientSGDHandler(ClientBackendHandler):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module):
        data_loader: torch.Dataloader for this client
        optimizer: optimizer for this client's model
        criterion: loss function used in local training process
        cuda (bool): use GPUs or not

    Returns:
        None

    Raises:
        None
    """
    def __init__(self, model, data_loader, optimizer=None, criterion=None, cuda=False):
        super(self, ClientSGDHandler).__init__(model, cuda)


        #self._buff = ravel_model_params(model, cuda=cuda)
        self._data_loader = data_loader
        self.cuda = cuda

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
        Client train its local model based on local dataset.

        Args:
            epochs (int): the number of epoch for local train
        """
        def accuracy_score(predicted, labels):
            return predicted.eq(labels).sum().float() / labels.shape[0]

        output_dir = "client_log.txt"
        train_recorder = ExRecorder(
            os.path.join(output_dir, "train_history_rank_%d" % self.args.local_rank))

        print("start train process...")
        for epoch in range(epochs):
            self._model.train()
            for inputs, labels in self._data_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                # 可优化data_loader从而优化数据转移过程
                self.optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(predicted, labels)

            train_recorder.add_log_direct({'epoch': epoch,
                                           'time': time.time(),
                                           'training_loss': loss.detach().item(),
                                           'training_accuracy': accuracy.item()})

        self._buff = ravel_model_params(self._model, cuda=True)

    def evaluate(self, test_loader):
        # TODO: Evaluate local model based on given test `torch.DataLoader`
        raise NotImplementedError()
