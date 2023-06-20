import argparse
import sys
from typing import Any

import torch

sys.path.append("../../")
torch.manual_seed(0)
from fedlab.board import fedboard
from fedlab.board.delegate import FedBoardDelegate
from fedlab.board.fedboard import RuntimeFedBoard
from handler import StandaloneSyncServerHandler
from pipeline import StandalonePipeline
from trainer import StandaloneSerialClientTrainer
from fedlab.models.mlp import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=50)
parser.add_argument("--com_round", type=int, default=1000)

parser.add_argument("--sample_ratio", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.01)

args = parser.parse_args()

model = MLP(784, 10)

# server
handler = StandaloneSyncServerHandler(model, args.com_round, args.sample_ratio)

# client
trainer = StandaloneSerialClientTrainer(model, args.total_client, cuda=False)
dataset = PathologicalMNIST(root='../../datasets/mnist/', path="../../datasets/mnist/", num_clients=args.total_client)
dataset.preprocess()
trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)


class mDelegate(FedBoardDelegate):
    def sample_client_data(self, client_id: str, type: str, amount: int) -> tuple[list[Any], list[Any]]:
        data = []
        label = []
        for dt in dataset.get_dataloader(client_id, batch_size=amount, type=type):
            x, y = dt
            for x_p in x:
                data.append(x_p)
            for y_p in y:
                label.append(y_p)
            break
        return data, label

    def read_client_label(self, client_id: str, type: str) -> list[Any]:
        res = []
        for _, label in dataset.get_dataloader(client_id, batch_size=args.batch_size, type=type):
            for y in label:
                res.append(y.detach().cpu().item())
        return res

# main
pipeline = StandalonePipeline(handler, trainer)
# set up FedLabBoard
fedboard.setup(mDelegate(), max_round=args.com_round, client_ids=[str(i) for i in range(args.total_client)])
# pipeline.main()
with RuntimeFedBoard(port=8040):
    pipeline.main()
