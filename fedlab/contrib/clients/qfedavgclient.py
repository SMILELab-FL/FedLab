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
import numpy as np

from .client import SGDClientTrainer


class qFedAvgClientTrainer(SGDClientTrainer):
    @property
    def uplink_package(self):
        return [self.delta, self.hk]

    def setup_optim(self, epochs, batch_size, lr, q):
        super().setup_optim(epochs, batch_size, lr)
        self.q = q
    
    def train(self, model_parameters, train_loader) -> None:
        """Client trains its local model on local dataset.
        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        self.set_model(model_parameters)
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            ret_loss = 0.0
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.gpu), target.cuda(
                        self.gpu)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                ret_loss += loss.detach().item()
        self._LOGGER.info("Local train procedure is finished")

        grad = (model_parameters - self.model_parameters) / self.lr
        self.delta = grad * np.float_power(ret_loss + 1e-10, self.q)
        self.hk = self.q * np.float_power(
            ret_loss + 1e-10, self.q - 1) * grad.norm(
            )**2 + 1.0 / self.lr * np.float_power(ret_loss + 1e-10, self.q)
