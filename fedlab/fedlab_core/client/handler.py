import os
from random import triangular
import time

import torch
from torch import nn
from fedlab_core.utils.serialization import ravel_model_params, unravel_model_params
from fedlab_core.utils.ex_recorder import ExRecorder


class ClientBackendHandler(object):
    """Abstract class"""

    def __init__(self) -> None:
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def update_model(self):
        pass

    def get_model(self):
        pass

    def get_buffer(self):
        pass

    def evaluate(self):
        pass


class ClientSGDHandler(ClientBackendHandler):
    def __init__(self, model, data_loader, args, optimizer=None, criterion=None, cuda=False):
        """client backend handler
           this class provides data process method to upper layer.  

            Args:
                model: torch.nn.Module
                data_loader: torch.Dataloader for this client
                optimizer: optimizer for this client's model
                criterion: loss function used in local training process
                cuda: use GPUs or not

            Returns: 
                None

            Raises:
                None
        """
        if cuda:
            self._model = model.cuda()
        else:
            self._model = model.cpu()

        self._buff = ravel_model_params(model, cuda=cuda)
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

        self.args = args

    def train(self, epochs):
        """
        client train its local model based on local dataset.
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

    def update_model(self, buff):
        """update local model using serialized parameters"""
        unravel_model_params(self._model, buff)
        self._buff[:] = buff

    def get_buff(self):
        """get serizlized parameters"""
        return self._buff

    def get_model(self):
        """get torch.nn.Module"""
        return self._model

    def evaluate(self, test_loader):
        """evaluate local model based on given test torch.DataLoader"""
        raise NotImplementedError()
