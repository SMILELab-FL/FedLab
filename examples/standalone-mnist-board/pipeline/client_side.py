from tqdm import tqdm

from fedlab.contrib.algorithm import SGDSerialClientTrainer


class ExampleClientTrainer(SGDSerialClientTrainer):

    def local_process(self, payload, id_list):
        self.loss = {}
        model_parameters = payload[0]
        for id in tqdm(id_list, desc=">>> Local training"):
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack, loss = self.train(model_parameters, data_loader)
            self.cache.append(pack)
            self.loss[id] = loss

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self._model.train()
        total_loss = 0
        for _ in range(self.epochs):
            total_loss = 0
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach().cpu().item()
        return [self.model_parameters], total_loss

    def get_loss(self):
        return self.loss
