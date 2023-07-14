import os
import pickle
import re

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import Dash, callback
from dash.dependencies import Output, Input, State, ALL
from dash.exceptions import PreventUpdate

from fedlab.board.front.layout import card_state, cyto_graph, card_overall_performance, page_performance, selection, \
    build_normal_charts, build_slider_charts
from fedlab.board.front.view_model import ViewModel
from fedlab.board.utils.io import _read_meta_file

viewModel = ViewModel()
_section_types: dict[str:str] = {}
_charts: dict[str:dict[str, dict]] = {}


def create_app(log_dir):
    viewModel.init(log_dir)
    app = Dash(__name__, title="FedBoard", update_title=None, assets_url_path='assets')
    app._favicon = 'favicon.png'
    return app


def _add_chart(section=None, figure_name=None, span=0.5):
    def ac(func):
        _charts.setdefault(section, {})
        _charts[section][figure_name] = {'func': func, 'name': figure_name, 'span': int(12 * span)}
        return func

    return ac


def _add_section(section: str, type: str):
    _section_types[section] = type


def get_selected_clients(selected_client, regex):
    try:
        regex = re.compile(regex)
        selected_client = [item for item in selected_client if regex.search(str(item))]
    except Exception as e:
        selected_client = []
    return selected_client


def add_dynamic_callback_normal(app, section, figure_id):
    @app.callback(
        Output(figure_id, "figure"),
        Input("client_selection_ms", "value"),
        Input("client_selection_reg", "value"),
        State(figure_id, "id"),
        background=True,
        manager=viewModel.background_callback_manager,
    )
    def wrapper(selected_client, regex, fig_id):
        selected_client = get_selected_clients(selected_client, regex)
        for sec, dic in _charts.items():
            if fig_id in dic.keys():
                fn = viewModel.encode_client_ids(selected_client)
                cached_path = os.path.join(viewModel.dir, f'cache/{fig_id}/')
                cached_file = os.path.join(cached_path, f'{fn}.pkl')
                os.makedirs(cached_path, exist_ok=True)
                if os.path.exists(cached_file):
                    return pickle.load(open(cached_file, 'rb'))
                selected_colors = [viewModel.get_color(id) for id in selected_client]
                fig = dic[fig_id]['func'](selected_client, selected_colors, viewModel.client_ids2ranks(selected_client))
                pickle.dump(fig, open(cached_file, 'wb'))
                return fig
        return None


def add_dynamic_callback_slider(app, section, figure_id):
    @app.callback(
        Output(figure_id, "figure"),
        Input({"type": "round_slider", "section": section}, "value"),
        Input("client_selection_ms", "value"),
        Input("client_selection_reg", "value"),
        State(figure_id, "id"),
        background=True,
        manager=viewModel.background_callback_manager,
    )
    def wrapper(value, selected_client, regex, fig_id):
        selected_client = get_selected_clients(selected_client, regex)
        for sec, dic in _charts.items():
            if fig_id in dic.keys():
                fn = viewModel.encode_client_ids(selected_client)
                cached_path = os.path.join(viewModel.dir, f'cache/{fig_id}/{value}/')
                cached_file = os.path.join(cached_path, f'{fn}.pkl')
                os.makedirs(cached_path, exist_ok=True)
                if os.path.exists(cached_file):
                    return pickle.load(open(cached_file, 'rb'))
                selected_colors = [viewModel.get_color(id) for id in selected_client]
                fig = dic[fig_id]['func'](value, selected_client, selected_colors,
                                          viewModel.client_ids2ranks(selected_client))
                pickle.dump(fig, open(cached_file, 'wb'))
                return fig
        return None


def _set_up_layout(app: Dash):
    tabs = [dmc.Tab('performance', value='performance', style={"font-size": 17})]
    for sec in _charts.keys():
        tabs.append(dmc.Tab(sec, value=sec, style={"font-size": 17}))
    tablist = dmc.TabsList(tabs)
    tabs_pages = [tablist, dmc.TabsPanel(page_performance, value="performance")]
    for section, type in _section_types.items():
        page = None
        if type == 'normal':
            page = build_normal_charts(section, _charts)
            for id, v in _charts[section].items():
                add_dynamic_callback_normal(app, section, v['name'])
        elif type == 'slider':
            page = build_slider_charts(section, _charts)
            for id, v in _charts[section].items():
                add_dynamic_callback_slider(app, section, v['name'])
        tabs_pages.append(dmc.TabsPanel(page, value=section))
    tabs = dmc.Tabs(tabs_pages, value="performance", variant='outline')
    bottom_page = dmc.Card(withBorder=True,
                           shadow="sm",
                           mt='sm',
                           radius="lg",
                           style={"width": '100%'},
                           children=[
                               dmc.Grid(
                                   children=[
                                       dmc.Col(selection, span='content'),
                                       dmc.Divider(orientation='vertical', mt='md', mb='md', mr='lg'),
                                       dmc.Col(tabs, span='auto', ml='xs')]
                               )
                           ])

    main = dmc.Grid([dmc.Col(card_state, span='content'), dmc.Col(
        cyto_graph, span='auto'), dmc.Col(card_overall_performance, span=5)
                        , dmc.Col(bottom_page, span=12)])
    app.layout = dmc.Container(
        [dmc.Header(height=110, children=[dmc.Grid(
            dmc.Col(dmc.Image(src='assets/FedLab-logo.svg', height=64, fit="contain", mt='lg'),
                    span=3))], style={"backgroundColor": "#ffffff"}, mb='lg'
                    ), main], fluid=True)


def _add_callbacks(app: Dash):
    @app.callback(
        Output("cytoscape", "elements"),
        Output("cytoscape", "layout"),
        Output("cytoscape", "stylesheet"),
        Input("select_cyto_layout", "value"))
    def update_layout(type):
        e, ss = viewModel.get_graph()
        layout = {}
        if type == 'cose':
            layout = {
                'idealEdgeLength': 690,
                'refresh': 20,
                'fit': True,
                'padding': 20,
                'randomize': False,
                'animate': True,
                'componentSpacing': 200,
                'nodeRepulsion': 100,
                'nodeOverlap': 1000,
                'edgeElasticity': 100,
                'nestingFactor': 24,
                'gravity': 200,
                'numIter': 1000,
                'initialTemp': 900,
                'coolingFactor': 0.99,
                'minTemp': 1.0
            }
        layout["name"] = type
        return e, layout, ss

    @app.callback(Output("client_selection_ms", "children"),
                  Input("client_selection_reg", "value"))
    def update_selection_list(regex):
        try:
            regex = re.compile(regex)
            selected_client = [item for item in viewModel.get_client_ids() if regex.search(str(item))]
        except Exception as e:
            selected_client = []
        cs = []
        for id in selected_client:
            nm = 'Clt' if viewModel.get_client_num() > 10 else 'Client'
            cs.append(dmc.Chip(
                f'{nm}{id}',
                value=id,
                variant="filled",
                size='xs'
            ))
        return cs

    @app.callback(Output("figure_overall", "figure"),
                  Output("name_overall", 'children'),
                  Input('select_overall_metrics', 'value'))
    def select_metric_to_figure(selected_metric):
        overall_perform, main_name = viewModel.get_overall_performance()
        if selected_metric == 'main':
            selected_metric = main_name
        performance = [obj[selected_metric] for round, obj in overall_perform]
        rounds = [round for round, obj in overall_perform]
        overall_perform_figure = go.Figure(data=go.Scatter(
            x=rounds, y=performance,
            mode='lines+markers'))
        overall_perform_figure.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            paper_bgcolor="white",
        )
        overall_perform_title = f'Overall Performance: {(performance[-1] if len(performance) > 0 else 0):.2f}'
        # metric_names = []
        # if len(overall_perform) > 0:
        #     for metric in overall_perform[-1][1].keys():
        #         if metric != 'main_name':
        #             metric_names.append(metric)
        return overall_perform_figure, overall_perform_title

    @app.callback(Output("figure_client_perform", "figure"),
                  Input('select_client_metrics', 'value'),
                  Input("client_selection_ms", "value"),
                  Input("client_selection_reg", "value"),
                  )
    def select_client_metric_to_figure(selected_metric, selected_clients, regex):
        selected_clients = get_selected_clients(selected_clients, regex)
        performs, main_name = viewModel.get_client_performance(selected_clients)
        if selected_metric == 'main':
            selected_metric = main_name
        figures = []
        for client_id, rd_pf in performs.items():
            performance = [obj[selected_metric] for _, obj in rd_pf]
            rounds = [rd for rd, obj in rd_pf]
            figures.append(go.Scatter(
                x=rounds, y=performance,
                mode='lines+markers',
                marker=dict(color=viewModel.get_color(client_id), size=10),
                name=f'Client{client_id}'))
        figure = go.Figure(data=figures)
        figure.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            paper_bgcolor="white",
        )
        return figure

    @app.callback(Output('current_state', 'children'),
                  Output('current_state', 'color'),
                  Output('round', 'children'),
                  Output('exp_progress', 'sections'),
                  Output('exp_progress_txt', 'children'),
                  Output("client_num", "children"),
                  Output("select_overall_metrics", 'data'),
                  Output("select_overall_metrics", 'value'),
                  Output("select_client_metrics", 'data'),
                  Output("select_client_metrics", 'value'),
                  Input('interval-component', 'n_intervals'),
                  State('select_overall_metrics', 'value'),
                  State('select_client_metrics', 'value'))
    def update_state(n, selected_overall_metric, selected_client_metric):
        states = _read_meta_file(viewModel.dir, 'runtime', ['state', 'round'])
        state = states['state']
        if state == 'RUNNING':
            color = 'blue'
        elif state == 'DONE':
            color = 'green'
        else:
            color = 'yellow'
        progress = 100 * float(states["round"]) / viewModel.get_max_round()
        metric_names, _ = viewModel.get_overall_metrics()
        client_metric_names, _ = viewModel.get_client_metrics()
        metric_names.append("main")
        client_metric_names.append("main")
        current_round = int(states['round'])
        return state, \
               color, \
               f'{current_round}/{viewModel.get_max_round()}', \
               [{"value": progress, "color": "indigo"}], \
               f'{progress:.1f}%', \
               viewModel.get_client_num(), \
               metric_names, \
               selected_overall_metric, \
               client_metric_names, \
               selected_client_metric

    @callback(Output("client_selection_ms", "value"),
              Input("client_selection_check", "n_clicks"),
              State("client_selection_ms", "value"),
              State("client_selection_ms", "children"), config_prevent_initial_callbacks=True)
    def select_all(n, selected, all):
        showed = set()
        for chip in all:
            showed.add(chip['props']['value'])
        all_selected = True
        for sid in showed:
            if sid not in selected:
                all_selected = False
                break
        selected = set(selected)
        if all_selected:
            selected = selected - showed
        else:
            selected = selected.union(showed)
        return list(selected)

    @app.callback(Output({'type': 'round_slider', 'section': ALL}, "max"),
                  Output({'type': 'round_slider', 'section': ALL}, "marks"),
                  Input('interval-component', 'n_intervals'))
    def update_state(n):
        states = _read_meta_file(viewModel.dir, 'runtime', ['state', 'round'])
        current_round = int(states['round'])
        slider_step = current_round // 5
        if slider_step == 0:
            slider_markers = [{"value": i, "label": f"{i}"} for i in range(1, current_round + 1)]
        else:
            slider_markers = [{"value": i, "label": f"{i}"} for i in range(1, current_round + 1, slider_step)]
        return [current_round], [slider_markers]
