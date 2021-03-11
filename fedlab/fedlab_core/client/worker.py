import os

import torch
from torch import nn
from fedlab.fedlab_core.utils.serialization import ravel_model_params, unravel_model_params


def ClientWork(self):
    def __init__(self, model, data_loader, optimizer, criterion, args):
        """

        """
        self._model = model
        self._buff = ravel_model_params(model)
        self._data_loader = data_loader

        self.optimizer = torch.optim.SGD(
            self._model.parameters(), lr=args.lr, momentum=args.momentum)
        # self.optimizer = optimizer
        # self.criterion = criterion
        self.criterion = nn.CrossEntropyLoss()

        self.args = args

    def train(self, args=self.args):

        for epoch in range(args.epochs):
            for i, data in enumerate(self._data_loader, 0):
                input, label = data

                if args.cuda:
                    input, label = input.cuda(), label.cuda()

                self.optimizer.zero_grad()
                output = self._model(input)
                loss = self.criterion(output, input)
                loss.backward()
                self.optimizer.step()

        self._buff = ravel_model_params(self._model)

    def update_model(self, buff):
        unravel_model_params(self._model, buff)
        self._buff[:] = buff

    def get_buff(self):
        return self._buff

    def get_model(self):
        return self._model

    def evaluate(self, test_loader):
        raise NotImplementedError()
