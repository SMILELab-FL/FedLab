from copy import deepcopy

import torch
from tqdm import tqdm

from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.utils import Logger


class StandaloneSerialClientTrainer(SerialClientTrainer):

    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.cache = []
        self.loss = {}

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size.
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in tqdm(id_list, desc=">>> Local training"):
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack, loss = self.train(model_parameters, data_loader)
            self.cache.append(pack)
            self.loss[id] = loss

    def train(self, model_parameters, train_loader):

        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            total_loss = 0
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach().cpu().item()
        return [self.model_parameters], total_loss

    def get_loss(self):
        return self.loss
