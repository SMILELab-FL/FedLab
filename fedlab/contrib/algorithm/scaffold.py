import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from ...utils import Aggregators

##################
#
#      Server
#
##################


class ScaffoldServerHandler(SyncServerHandler):
    """FedAvg server handler."""
    @property
    def downlink_package(self):
        return [self.model_parameters, self.global_c]

    def setup_optim(self, lr):
        self.lr = lr
        self.global_c = torch.zeros_like(self.model_parameters)

    def global_update(self, buffer):
        # unpack
        dys = [ele[0] for ele in buffer]
        dcs = [ele[1] for ele in buffer]

        dx = Aggregators.fedavg_aggregate(dys)
        dc = Aggregators.fedavg_aggregate(dcs)

        next_model = self.model_parameters + self.lr * dx
        self.set_model(next_model)

        self.global_c += 1.0 * len(dcs) / self.num_clients * dc


##################
#
#      Client
#
##################


class ScaffoldSerialClientTrainer(SGDSerialClientTrainer):
    def setup_optim(self, epochs, batch_size, lr):
        super().setup_optim(epochs, batch_size, lr)
        self.cs = [None for _ in range(self.num_clients)]

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        global_c = payload[1]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(id, model_parameters, global_c, data_loader)
            self.cache.append(pack)

    def train(self, id, model_parameters, global_c, train_loader):
        self.set_model(model_parameters)
        frz_model = model_parameters

        if self.cs[id] is None:
            self.cs[id] = torch.zeros_like(model_parameters)

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()

                grad = self.model_gradients
                grad = grad - self.cs[id] + global_c
                idx = 0
                for parameter in self._model.parameters():
                    layer_size = parameter.grad.numel()
                    shape = parameter.grad.shape
                    #parameter.grad = parameter.grad - self.cs[id][idx:idx + layer_size].view(parameter.grad.shape) + global_c[idx:idx + layer_size].view(parameter.grad.shape)
                    parameter.grad.data[:] = grad[idx:idx+layer_size].view(shape)[:]
                    idx += layer_size

                self.optimizer.step()

        dy = self.model_parameters - frz_model
        dc = -1.0 / (self.epochs * len(train_loader) * self.lr) * dy - global_c
        self.cs[id] += dc
        return [dy, dc]
