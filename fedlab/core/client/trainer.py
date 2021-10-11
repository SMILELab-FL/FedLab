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

from tqdm import tqdm
from ..client import ORDINARY_TRAINER
from ...utils import Logger
from ...utils.serialization import SerializationTool
from ..model_maintainer import ModelMaintainer


class ClientTrainer(ModelMaintainer):
    """An abstract class representing a client backend trainer.

    In our framework, we define the backend of client trainer show manage its local model.
    It should have a function to update its model called :meth:`train`.

    If you use our framework to define the activities of client, please make sure that your self-defined class
    should subclass it. All subclasses should overwrite :meth:`train`.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool): Use GPUs or not.
    """

    def __init__(self, model, cuda):
        super().__init__(model, cuda)
        self.client_num = 1  # default is 1.
        self.type = ORDINARY_TRAINER

    def train(self):
        """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate quality of local model."""
        raise NotImplementedError()


class ClientSGDTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        data_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        epochs (int): the number of local epoch.
        optimizer (torch.optim.Optimizer, optional): optimizer for this client's model.
        criterion (torch.nn.Loss, optional): loss function used in local training process.
        cuda (bool, optional): use GPUs or not. Default: ``True``.
        logger (Logger, optional): :object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=True,
                 logger=Logger()):
        super(ClientSGDTrainer, self).__init__(model, cuda)

        self._data_loader = data_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self._LOGGER = logger

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
            for inputs, labels in tqdm(self._data_loader,
                                       desc="{}, Epoch {}".format(self._LOGGER.name, ep)):
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(
                        self.gpu)

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")
        return self.model_parameters
