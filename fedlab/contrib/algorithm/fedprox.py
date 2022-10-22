from copy import deepcopy
import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer


##################
#
#      Server
#
##################


class FedProxServerHandler(SyncServerHandler):
    """FedProx server handler."""
    None


##################
#
#      Client
#
##################

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
        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(
                        self.device)

                preds = self._model(data)
                l1 = self.criterion(preds, target)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * mu * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return [self.model_parameters]

class FedProxSerialClientTrainer(SGDSerialClientTrainer):
    def setup_optim(self, epochs, batch_size, lr, mu):
        super().setup_optim(epochs, batch_size, lr)
        self.mu = mu

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader, self.mu)
            self.cache.append(pack)

    def train(self, model_parameters, train_loader, mu) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        self.set_model(model_parameters)
        frz_model = deepcopy(self._model)
        frz_model.eval()

        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(
                        self.device)

                preds = self._model(data)
                l1 = self.criterion(preds, target)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * mu * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]
