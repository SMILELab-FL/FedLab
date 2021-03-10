import os

from fedlab_core.utils.serialization import ravel_model_params


def ClientWork(self):
    def __init__(self, model, data_loader, args):
        """
        
        """
        self._model = model
        self._buff = ravel_model_params(model)
        self._data_loader = data_loader

    def train(self, buff):
        raise NotImplementedError()

    def get_buff(self):
        return self._buff

    def get_model(self):
        return self._model

    def evaluate(self, test_loader):
        raise NotImplementedError()
