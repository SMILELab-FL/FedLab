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

from .client import SGDClientTrainer


class FedProxClientTrainer(SGDClientTrainer):
    """Federated client with local SGD with proximal term solver."""
    def setup_optim(self, epochs, batch_size, lr, mu):
        super().setup_optim(epochs, batch_size, lr)
        self.mu = mu

    def local_process(self, payload, id):
        model_parameters = payload[0]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader, self.mu)

    def train(self, model_parameters, train_loader, mu) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        self.set_model(model_parameters)
        frz_model = deepcopy(self._model)
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.gpu), target.cuda(
                        self.gpu)

                preds = self._model(data)
                l1 = self.criterion(preds, target)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * mu * l2
                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")