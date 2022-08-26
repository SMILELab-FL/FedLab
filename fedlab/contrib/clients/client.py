# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
import torch
from ...core.client.trainer import ClientTrainer, SerialClientTrainer
from ...utils import Logger, SerializationTool

class SGDClientTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        epochs (int): the number of local epoch.
        optimizer (torch.optim.Optimizer): optimizer for this client's model.
        criterion (torch.nn.Loss): loss function used in local training process.
        cuda (bool, optional): use GPUs or not. Default: ``False``.
        logger (Logger, optional): :object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 cuda=False,
                 device=None,
                 logger=None):
        super(SGDClientTrainer, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        return [self.model_parameters]

    def setup_dataset(self, dataset):
        self.dataset = dataset
    
    def setup_optim(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_process(self, payload, id):
        model_parameters = payload[0]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)

    def train(self, model_parameters, train_loader) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")


class SGDSerialTrainer(SerialClientTrainer):
    """Deprecated
    Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        logger (Logger, optional): Object of :class:`Logger`.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str):
        args (dict): Uncertain variables. Default: ``{"epochs": 5, "batch_size": 100, "lr": 0.1}``

    .. note::
        ``len(data_slices) == client_num``, that is, each sub-index of :attr:`dataset` corresponds to a client's local dataset one-by-one.
    """

    def __init__(self, model, num, logger=None, cuda=False, device=None, personal=False) -> None:
        super().__init__(model, num, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.chache = []

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    @property
    def uplink_package(self):
        package = deepcopy(self.chache)
        self.chache = []
        return package
    
    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader)
            self.chache.append(pack)

    def train(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]