import numpy as np

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer


##################
#
#      Server
#
##################


class qFedAvgServerHandler(SyncServerHandler):
    """qFedAvg server handler."""
    def global_update(self, buffer):
        deltas = [ele[0] for ele in buffer]
        hks = [ele[1] for ele in buffer]

        hk = sum(hks)
        updates = sum([delta/hk for delta in deltas])
        model_parameters = self.model_parameters - updates

        self.set_model(model_parameters)


##################
#
#      Client
#
##################


class qFedAvgClientTrainer(SGDClientTrainer):
    """Federated client with modified upload package and local SGD solver."""
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
