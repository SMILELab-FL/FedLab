import argparse
import sys
from typing import Any

sys.path.append("../../")
from fedlab.board import fedboard
from fedlab.board.delegate import FedBoardDelegate
from fedlab.board.utils.roles import BOARD_SHOWER
from fedlab.contrib.dataset import PathologicalMNIST

parser = argparse.ArgumentParser(description="FedBoard example")
parser.add_argument("--port", type=str, default="8070")
args = parser.parse_args()

fedboard.register(id='mtp-01', roles=BOARD_SHOWER)
dataset = PathologicalMNIST(root='../../datasets/mnist/',
                            path="../../datasets/mnist/",
                            num_clients=100)


class mDelegate(FedBoardDelegate):
    def sample_client_data(self, client_id: str, client_rank: str, type: str, amount: int) -> tuple[
        list[Any], list[Any]]:
        data = []
        label = []
        real_id = int(client_rank) * 10 + int(client_id.split('-')[-1])
        for dt in dataset.get_dataloader(real_id, batch_size=amount, type=type):
            x, y = dt
            for x_p in x:
                data.append(x_p)
            for y_p in y:
                label.append(y_p)
            break
        return data, label

    def read_client_label(self, client_id: str, client_rank: str, type: str) -> list[Any]:
        res = []
        real_id = int(client_rank) * 10 + int(client_id.split('-')[1])
        for _, label in dataset.get_dataloader(real_id, batch_size=64, type=type):
            for y in label:
                res.append(y.detach().cpu().item())
        return res


fedboard.enable_builtin_charts(mDelegate())
fedboard.start(port=args.port)
