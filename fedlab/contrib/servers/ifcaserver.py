
import torch

from .server import SyncServerHandler
from ...utils.aggregator import Aggregators

class IFCAServerHander(SyncServerHandler):
    def __init__(self, model: torch.nn.Module, global_round: int, sample_ratio: float, cuda: bool = False, device: str = None, logger = None):
        super().__init__(model, global_round, sample_ratio, cuda, device, logger)
    
    @property
    def downlink_package(self):
        return [self.shared_paramters] + self.global_models
    
    def setup_optim(self, share_size, k, init_parameters):
        """_summary_

        Args:
            share_size (_type_): _description_
            k (_type_): _description_
            init_parameters (_type_): _description_
        """
        assert k == len(init_parameters)
        self.k = k
        self.share_size = share_size

        self.global_models = init_parameters
        self.shared_paramters = Aggregators.fedavg_aggregate(self.global_models)[0:self.share_size]

    def global_update(self, buffer):
        cluster_model = [[] for _ in range(self.k)]
        # weights = [[] for _ in range(self.k)]
        for i, (cid, id, paramters) in enumerate(buffer):
            cluster_model[cid].append(paramters)
            # weights[cid].append(self.client_trainer.weights[id])

        parameters = Aggregators.fedavg_aggregate([ele for _, _, ele in buffer])
        self.shared_paramters[0:self.share_size] = parameters[0:self.share_size]

        for i, ele in enumerate(cluster_model):
            if len(ele) > 0:
                self.global_models[i] = Aggregators.fedavg_aggregate(ele)
