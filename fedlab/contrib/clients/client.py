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

import torch
from ...core.client.trainer import ClientTrainer, SerialTrainer
from ...utils import Logger, SerializationTool
from ...utils.dataset import SubsetSampler

class SGDClientTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        data_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        epochs (int): the number of local epoch.
        optimizer (torch.optim.Optimizer): optimizer for this client's model.
        criterion (torch.nn.Loss): loss function used in local training process.
        cuda (bool, optional): use GPUs or not. Default: ``False``.
        logger (Logger, optional): :object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=False,
                 logger=None):
        super(SGDClientTrainer, self).__init__(model, cuda)

        self._data_loader = data_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self._LOGGER = Logger() if logger is None else logger

        self.model_time = 0

    @property
    def uplink_package(self):
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        return [self.model_parameters]

    def local_process(self, payload):
        model_parameters = payload[0]
        self.train(model_parameters)

    def train(self, model_parameters) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for inputs, labels in self._data_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(
                        self.gpu)

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")


class SubsetSerialTrainer(SerialTrainer):
    """Deprecated
    Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        dataset (torch.utils.data.Dataset): Local dataset for this group of clients.
        data_slices (list[list]): Subset of indices of dataset.
        logger (Logger, optional): Object of :class:`Logger`.
        cuda (bool): Use GPUs or not. Default: ``False``.
        args (dict): Uncertain variables. Default: ``{"epochs": 5, "batch_size": 100, "lr": 0.1}``

    .. note::
        ``len(data_slices) == client_num``, that is, each sub-index of :attr:`dataset` corresponds to a client's local dataset one-by-one.
    """

    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 logger=None,
                 cuda=False,
                 args={
                     "epochs": 5,
                     "batch_size": 100,
                     "lr": 0.1
                 }) -> None:
        print("SubsetSerialTrainer would be deprecated in the next version.")
        super(SubsetSerialTrainer, self).__init__(model=model,
                                                  client_num=len(data_slices),
                                                  cuda=cuda,
                                                  logger=logger)

        self.dataset = dataset
        self.data_slices = data_slices  # [0, client_num)
        self.args = args

    def _get_dataloader(self, client_id):
        """Return a training dataloader used in :meth:`train` for client with :attr:`id`

        Args:
            client_id (int): :attr:`client_id` of client to generate dataloader

        Note:
            :attr:`client_id` here is not equal to ``client_id`` in global FL setting. It is the index of client in current :class:`SerialTrainer`.

        Returns:
            :class:`DataLoader` for specific client's sub-dataset
        """
        batch_size = self.args["batch_size"]

        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[client_id],
                                  shuffle=True),
            batch_size=batch_size)

        return train_loader

    def _train_alone(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        epochs, lr = self.args["epochs"], self.args["lr"]
        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._model.train()

        for _ in range(epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                output = self.model(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model_parameters