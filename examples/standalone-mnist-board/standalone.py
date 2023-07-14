import argparse
import sys

import torch
from sklearn.manifold import TSNE

from fedlab.contrib.algorithm import SyncServerHandler

sys.path.append("../../")
torch.manual_seed(0)
from fedlab.board import fedboard
from fedlab.board.fedboard import RuntimeFedBoard
from pipeline.server_side import ExamplePipeline, ExampleHandler
from pipeline.client_side import ExampleClientTrainer
from fedlab.models.mlp import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
import plotly.graph_objects as go
from fedlab.board.utils.roles import ALL
from board import ExampleDelegate

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=50)
parser.add_argument("--com_round", type=int, default=1000)

parser.add_argument("--sample_ratio", type=float, default=0.5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--port", type=int, default=8040)

args = parser.parse_args()

model = MLP(784, 10)

# server
handler = ExampleHandler(model, args.com_round, args.sample_ratio)

# client
trainer = ExampleClientTrainer(model, args.total_client, cuda=False)
dataset = PathologicalMNIST(root='../../datasets/mnist/', path="../../datasets/mnist/", num_clients=args.total_client)
dataset.preprocess()
trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# main
pipeline = ExamplePipeline(handler, trainer)

# set up FedLabBoard
fedboard.register(max_round=args.com_round, roles=ALL,
                  client_ids=[str(i) for i in range(args.total_client)])

# To enable builtin figures, a dataset-reading delegate is required
delegate = ExampleDelegate(dataset)
fedboard.enable_builtin_charts(delegate)

# Add diy chart
fedboard.add_section(section='diy', type='normal')


@fedboard.add_chart(section='diy', figure_name='2d-dataset-tsne', span=1.0)
def diy_chart(selected_clients, selected_colors, selected_rank):
    """
    Args:
        selected_clients: selected client ids, ['1','2',...'124']
        selected_colors: colors of selected clients, ['#ffffff','#982223',...,'#128842']
    Returns:
        A Plotly Figure
    """
    raw = []
    client_range = {}
    for client_id, rank in zip(selected_clients, selected_rank):
        data, label = delegate.sample_client_data(client_id, rank, 'train', 100)
        client_range[client_id] = (len(raw), len(raw) + len(data))
        raw += data
    raw = torch.stack(raw).view(len(raw), -1)
    tsne = TSNE(n_components=2, learning_rate=100, random_state=501,
                perplexity=min(30.0, len(raw) - 1)).fit_transform(raw)
    tsne_data = {cid: tsne[s:e] for cid, (s, e) in client_range.items()}
    data = []
    for idx, cid in enumerate(selected_clients):
        data.append(go.Scatter(
            x=tsne_data[cid][:, 0], y=tsne_data[cid][:, 1], mode='markers',
            marker=dict(color=selected_colors[idx], size=8, opacity=0.8),
            name=f'Client{cid}'
        ))
    tsne_figure = go.Figure(data=data,
                            layout_title_text=f"Local Dataset 2D t-SNE")
    tsne_figure.update_layout(margin=dict(l=48, r=48, b=64, t=80), dict1={"height": 600})
    return tsne_figure


# Start experiment along with FedBoard
with RuntimeFedBoard(port=args.port):
    pipeline.main()
