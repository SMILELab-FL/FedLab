import os
from random import triangular
import time

import torch
from torch import nn
from fedlab_core.utils.serialization import ravel_model_params, unravel_model_params
from fedlab_core.utils.ex_recorder import ExRecorder


class ClientBackendHandler(object):
    """
    抽象类
    """
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


class ClientSGDHandler:
    def __init__(self, model, data_loader, args, optimizer=None, criterion=None):
        """
        client
        后端工作
        """
        self._model = model.cuda()
        self._buff = ravel_model_params(model, cuda=True)
        self._data_loader = data_loader

        if optimizer is None:
            self.optimizer = torch.optim.SGD(
                self._model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            self.optimizer = optimizer
        
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        self.args = args

    def train(self, args):

        def accuracy_score(predicted, labels):
            return predicted.eq(labels).sum().float() / labels.shape[0]

        output_dir = "client_log.txt"
        train_recorder = ExRecorder(
            os.path.join(output_dir, "train_history_rank_%d" % args.local_rank))

        for epoch in range(args.epochs):
            self._model.train()
            for inputs, labels in self._data_loader:
                if args.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

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
        unravel_model_params(self._model, buff)
        self._buff[:] = buff

    def get_buff(self):
        return self._buff

    def get_model(self):
        return self._model

    def evaluate(self, test_loader):
        raise NotImplementedError()
