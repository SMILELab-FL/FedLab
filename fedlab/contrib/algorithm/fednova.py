import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from ...utils import Aggregators

##################
#
#      Server
#
##################


class FedNovaServerHandler(SyncServerHandler):
    """FedAvg server handler."""

    def setup_optim(self, option="weighted_scale"):
        self.option = option  # weighted_scale, uniform, weighted_com

    def global_update(self, buffer):
        models = [elem[0] for elem in buffer]
        taus = [elem[1] for elem in buffer]

        deltas = [(model - self.model_parameters)/tau for model, tau in zip(models, taus)]

        # p is the FedAvg weight, we simply set it 1/m here.
        p = [
            1.0 / self.num_clients_per_round
            for _ in range(self.num_clients_per_round)
        ]

        if self.option == 'weighted_scale':
            K = len(deltas)
            N = self.num_clients
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = sum([dk * pk
                                        for dk, pk in zip(deltas, p)]) * N / K

        elif self.option == 'uniform':
            tau_eff = 1.0 * sum(taus) / len(deltas)
            delta = Aggregators.fedavg_aggregate(deltas)

        elif self.option == 'weighted_com':
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = sum([dk * pk for dk, pk in zip(deltas, p)])

        else:
            sump = sum(p)
            p = [pk / sump for pk in p]
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = sum([dk * pk for dk, pk in zip(deltas, p)])

        self.set_model(self.model_parameters + tau_eff * delta)


##################
#
#      Client
#
##################


class FedNovaSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader)
            tau = [torch.Tensor([len(data_loader) * self.epochs])]
            pack += tau
            self.cache.append(pack)