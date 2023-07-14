import plotly.graph_objects as go

from fedlab.board import fedboard
from fedlab.board.builtin.renderer import client_param_tsne, get_client_dataset_tsne, get_client_data_report


def add_built_in_charts():
    fedboard.add_section('dataset', 'normal')
    fedboard.add_section('parameters', 'slider')

    @fedboard.add_chart(section='parameters', figure_name='figure_tsne', span=1.0)
    def update_tsne_figure(value, selected_client, selected_colors, selected_ranks):
        tsne_data, id_existed = client_param_tsne(value, selected_client)
        if tsne_data is not None:
            data = []
            for idx, cid in enumerate(id_existed):
                data.append(go.Scatter(
                    x=[tsne_data[idx, 0]], y=[tsne_data[idx, 1]], mode='markers',
                    marker=dict(color=selected_colors[idx], size=16),
                    name=f'Client{cid}'
                ))
            tsne_figure = go.Figure(data=data,
                                    layout_title_text=f"Parameters t-SNE")
        else:
            tsne_figure = []
        return tsne_figure

    @fedboard.add_chart(section='dataset', figure_name='figure_client_classes', span=0.5)
    def update_data_classes(selected_client, selected_colors, selected_ranks):
        client_targets = get_client_data_report(selected_client, 'train', selected_ranks)
        class_sizes: dict[str, dict[str, int]] = {}
        for cid, targets in client_targets.items():
            for y in targets:
                class_sizes.setdefault(y, {id: 0 for id in selected_client})
                class_sizes[y][cid] += 1
        client_classes = go.Figure(
            data=[
                go.Bar(y=[f'Client{id}' for id in selected_client],
                       x=[sizes[id] for id in selected_client],
                       name=f'Class {clz}', orientation='h')
                # marker=dict(color=[viewModel.colors[id] for id in selected_client])),
                for clz, sizes in class_sizes.items()
            ],
            layout_title_text="Label Distribution"
        )
        client_classes.update_layout(barmode='stack', margin=dict(l=48, r=48, b=64, t=86))
        return client_classes

    @fedboard.add_chart(section='dataset', figure_name='figure_client_sizes', span=0.5)
    def update_data_sizes(selected_client, selected_colors, selected_ranks):
        client_targets = get_client_data_report(selected_client, 'train', selected_ranks)
        client_sizes = go.Figure(
            data=[go.Bar(x=[f'Client{n}' for n, _ in client_targets.items()],
                         y=[len(ce) for _, ce in client_targets.items()],
                         marker=dict(color=selected_colors))],
            layout_title_text="Dataset Sizes"
        )
        client_sizes.update_layout(margin=dict(l=48, r=48, b=64, t=86))
        return client_sizes

    @fedboard.add_chart(section='dataset', figure_name='figure_client_data_tsne', span=1.0)
    def update_data_tsne_value(selected_client, selected_colors, selected_ranks):
        tsne_data = get_client_dataset_tsne(selected_client, "train", 200, selected_ranks)
        if tsne_data is not None:
            data = []
            for idx, cid in enumerate(selected_client):
                data.append(go.Scatter3d(
                    x=tsne_data[cid][:, 0], y=tsne_data[cid][:, 1], z=tsne_data[cid][:, 2], mode='markers',
                    marker=dict(color=selected_colors[idx], size=4, opacity=0.8),
                    name=f'Client{cid}'
                ))
        else:
            data = []
        tsne_figure = go.Figure(data=data,
                                layout_title_text=f"Local Dataset t-SNE")
        tsne_figure.update_layout(margin=dict(l=48, r=48, b=64, t=64), dict1={"height": 600})
        return tsne_figure
