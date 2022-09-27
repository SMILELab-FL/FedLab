import torch
import tqdm
import numpy as np

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from ...utils import SerializationTool, Aggregators
from ...utils.functional import evaluate


##################
#
#      Server
#
##################


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


##################
#
#      Client
#
##################


class IFCASerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)

    def setup_dataset(self, dataset):
        return super().setup_dataset(dataset)

    def setup_optim(self, epochs, batch_size, lr):
        return super().setup_optim(epochs, batch_size, lr)

    def local_process(self, payload, id_list):
        shared_model = payload[0]
        payload = payload[1:0]

        criterion = torch.nn.CrossEntropyLoss()
        results = []
        for id in tqdm(id_list):
            data_loader = self._get_dataloader(id, self.args.batch_size)
            if  len(payload) > 1:
                eval_loss = []
                for i, model_parameters in enumerate(payload):

                    model_parameters[0:shared_model.shape[0]] = shared_model[:]
                    payload[i] = model_parameters

                    SerializationTool.deserialize_model(self._model, model_parameters)
                    loss, _ = evaluate(self._model, criterion, data_loader)
                    eval_loss.append(loss)
                latent_cluster = np.argmin(eval_loss)
            else:
                latent_cluster = 0
            
            model_parameters = self.train(payload[latent_cluster], data_loader)
            results.append((latent_cluster, id, model_parameters))
