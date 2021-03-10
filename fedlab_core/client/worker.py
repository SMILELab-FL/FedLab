import os

import torch

from fedlab_core.utils.serialization import ravel_model_params, unravel_model_params


def ClientWork(self):
    def __init__(self, model, data_loader, args):
        """
        
        """
        self._model = model
        self._buff = ravel_model_params(model)
        self._data_loader = data_loader

    def train(self, buff):
        unravel_model_params(self._model, buff) # load model params
        optimizer = torch.optimizer

    def get_buff(self):
        return self._buff

    def get_model(self):
        return self._model

    def evaluate(self, test_loader):
        raise NotImplementedError()
