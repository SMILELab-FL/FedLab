
import torch
import numpy as np
import tqdm

from ...utils.functional import evaluate
from ...utils.serialization import SerializationTool
from .client import SGDSerialClientTrainer

class IFCAClientTrainer(SGDSerialClientTrainer):
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

    