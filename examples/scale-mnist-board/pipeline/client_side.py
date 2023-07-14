from tqdm import tqdm

from fedlab.board import fedboard
from fedlab.contrib.algorithm import SGDSerialClientTrainer


class ExampleTrainer(SGDSerialClientTrainer):

    def __init__(self, rank, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self._model.train()
        loss = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters, loss.detach().cpu()]

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        global_round = int(payload[-1].detach().cpu().item())
        client_metrics = {}
        client_parameters = {}
        for id in tqdm(id_list, desc=">>> Local training"):
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)
            global_id = f'{self.rank}-{id % 10}'
            client_parameters[global_id] = pack[0]
            loss = float(pack[-1].numpy())
            client_metrics[global_id] = {'loss': loss, 'nloss': -loss}

        fedboard.log(global_round, client_metrics=client_metrics, client_params=client_parameters)
