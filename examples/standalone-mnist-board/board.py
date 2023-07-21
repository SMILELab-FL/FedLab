from typing import Any

from fedlab.board.delegate import FedBoardDelegate


class ExampleDelegate(FedBoardDelegate):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def sample_client_data(self, client_id: str, client_rank: str, type: str, amount: int) -> tuple[
        list[Any], list[Any]]:
        data = []
        label = []
        for dt in self.dataset.get_dataloader(client_id, batch_size=amount, type=type):
            x, y = dt
            for x_p in x:
                data.append(x_p)
            for y_p in y:
                label.append(y_p)
            break
        return data, label

    def read_client_label(self, client_id: str, client_rank: str, type: str) -> list[Any]:
        res = []
        for _, label in self.dataset.get_dataloader(client_id, batch_size=64, type=type):
            for y in label:
                res.append(y.detach().cpu().item())
        return res
