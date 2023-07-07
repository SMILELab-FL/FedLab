import torch
import numpy as np
from .utils_algorithms import MinNormSolver

from .basic_server import SyncServerHandler
from ...utils.aggregator import Aggregators
from ...utils.serialization import SerializationTool



class FedMGDAServerHandler(SyncServerHandler):
    def setup_optim(self, sampler, lr):
        self.n = self.num_clients
        self.num_to_sample = int(self.sample_ratio * self.n)
        self.round_clients = int(self.sample_ratio * self.n)
        self.sampler = sampler

        self.lr = lr
        self.solver = MinNormSolver

    @property
    def num_clients_per_round(self):
        return self.round_clients

    def sample_clients(self, num_to_sample=None):
        clients = self.sampler.sample(self.num_to_sample)
        self.round_clients = len(clients)
        assert self.num_clients_per_round == len(clients)
        return clients

    def global_update(self, buffer):
        gradient_list = [
            torch.sub(self.model_parameters, ele[0]) for ele in buffer
        ]

        # MGDA+
        norms = np.array(
            [torch.norm(grad, p=2, dim=0).item() for grad in gradient_list])
        normlized_gradients = [
            grad / n for grad, n in zip(gradient_list, norms)
        ]
        sol, val = self.solver.find_min_norm_element_FW(normlized_gradients)
        print("GDA {}".format(val))
        assert val > 1e-5
        estimates = Aggregators.fedavg_aggregate(normlized_gradients, sol)

        serialized_parameters = self.model_parameters - self.lr * estimates
        SerializationTool.deserialize_model(self._model, serialized_parameters)
