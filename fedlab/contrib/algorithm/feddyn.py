import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from ...utils import Aggregators


##################
#
#      Server
#
##################


class FedDynServerHandler(SyncServerHandler):
    """FedAvg server handler."""
    def setup_optim(self, alpha):
        self.alpha = alpha
        self.h = torch.zeros_like(self.model_parameters)

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        self.h = self.h - self.alpha * (1.0/self.num_clients) * (sum(parameters_list) - self.model_parameters)
        new_parameters = Aggregators.fedavg_aggregate(parameters_list) - 1.0 / self.alpha * self.h
        self.set_model(new_parameters)


##################
#
#      Client
#
##################


class FedDynSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)

        self.L = [None for _ in range(num_clients)]


    def setup_dataset(self, dataset):
        return super().setup_dataset(dataset)

    def setup_optim(self, epochs, batch_size, lr, alpha):
        self.alpha = alpha
        super().setup_optim(epochs, batch_size, lr)

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(id, model_parameters, data_loader)
            self.cache.append(pack)

    def train(self, id, model_parameters, train_loader):
        if self.L[id] is None:
            self.L[id] = torch.zeros_like(model_parameters)

        L_t = self.L[id]
        frz_parameters = model_parameters

        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                l1 = self.criterion(output, target)
                l2 = torch.dot(L_t, self.model_parameters)
                l3 = torch.sum(torch.pow(self.model_parameters - frz_parameters,2))
                
                loss = l1 - l2 + 0.5 * self.alpha * l3

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.L[id] = L_t - self.alpha * (self.model_parameters-frz_parameters)

        return [self.model_parameters]
