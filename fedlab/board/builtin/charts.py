
from fedlab.board.front.app import viewModel, _add_section, _add_chart
import plotly.graph_objects as go


def _add_built_in_charts():
    _add_section('dataset', 'normal')
    _add_section('parameters', 'slider')

    @_add_chart(section='parameters', figure_name='figure_tsne', span=12)
    def update_tsne_figure(value, selected_client):
        tsne_data = viewModel.client_param_tsne(value, selected_client)
        if tsne_data is not None:
            data = []
            for idx, cid in enumerate(selected_client):
                data.append(go.Scatter(
                    x=[tsne_data[idx, 0]], y=[tsne_data[idx, 1]], mode='markers',
                    marker=dict(color=viewModel.get_color(cid), size=16),
                    name=f'Client{cid}'
                ))
            tsne_figure = go.Figure(data=data,
                                    layout_title_text=f"Parameters t-SNE")
        else:
            tsne_figure = []
        return tsne_figure

    @_add_chart(section='dataset', figure_name='figure_client_classes', span=6)
    def update_data_classes(selected_client):
        client_targets = viewModel.get_client_data_report(selected_client, type='train')
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

    @_add_chart(section='dataset', figure_name='figure_client_sizes', span=6)
    def update_data_sizes(selected_client):
        client_targets = viewModel.get_client_data_report(selected_client, type='train')
        client_sizes = go.Figure(
            data=[go.Bar(x=[f'Client{n}' for n, _ in client_targets.items()],
                         y=[len(ce) for _, ce in client_targets.items()],
                         marker=dict(color=[viewModel.get_color(id) for id in selected_client]))],
            layout_title_text="Dataset Sizes"
        )
        client_sizes.update_layout(margin=dict(l=48, r=48, b=64, t=86))
        return client_sizes

    @_add_chart(section='dataset', figure_name='figure_client_data_tsne', span=12)
    def update_data_data_value(selected_client):
        tsne_data = viewModel.get_client_dataset_tsne(selected_client, "train", 200)
        if tsne_data is not None:
            data = []
            for idx, cid in enumerate(selected_client):
                data.append(go.Scatter3d(
                    x=tsne_data[cid][:, 0], y=tsne_data[cid][:, 1], z=tsne_data[cid][:, 2], mode='markers',
                    marker=dict(color=viewModel.get_color(cid), size=4, opacity=0.8),
                    name=f'Client{cid}'
                ))
        else:
            data = []
        tsne_figure = go.Figure(data=data,
                                layout_title_text=f"Local Dataset t-SNE")
        tsne_figure.update_layout(margin=dict(l=48, r=48, b=64, t=64), dict1={"height": 600})
        return tsne_figure
