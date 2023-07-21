import dash_cytoscape as cyto
import dash_mantine_components as dmc
from dash import dcc
from dash import html
from dash_iconify import DashIconify

OVERVIEW_HEIGHT = 300
OVERVIEW_WIDTH = 290
card_state = dmc.Card(
    children=[
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        )
        , dmc.Center(dmc.RingProgress(
            id="exp_progress",
            sections=[{"value": 0, "color": "indigo"}],
            label=dmc.Center(dmc.Text("-%", color="indigo", id="exp_progress_txt", size=18)),
        ), ),
        dmc.Group(
            [
                dmc.Text("Experiment Status", size=14),
                # dmc.Text("N/A", id='current_state', weight=500),
                dmc.Badge("-", color="red", variant="light", id="current_state", size=14),
            ],
            position="apart",
            mt="md",
            mb="xs",
        ),
        dmc.Group(
            [
                dmc.Text("Communication round", size=14),
                dmc.Text("N/A", id='round', size=14),
            ],
            position="apart",
            mt="md",
            mb="xs",
        ),
        dmc.Group(
            [
                dmc.Text("Amount of Clients", size=14),
                dmc.Text("N/A", id='client_num', size=14),
            ],
            position="apart",
            mt="md",
            mb="xs",
        ),
    ],
    withBorder=True,
    shadow="lg",
    radius="lg",
    style={"width": OVERVIEW_WIDTH, 'height': OVERVIEW_HEIGHT},
)

card_overall_performance = dmc.Card(
    withBorder=True,
    shadow="lg",
    radius="lg",
    style={'height': OVERVIEW_HEIGHT},
    children=[
        dmc.Group(children=[
            dmc.Text("Overall Performance", id='name_overall', size=16),
            dmc.Select(id='select_overall_metrics', size='xs', clearable=False, value="main")]
            , position="apart"),
        dmc.Space(h='md'),
        dcc.Graph(id='figure_overall', style={"height": "80%"},
                  config={'autosizable': False, 'displaylogo': False})
    ]
)

cyto_graph = dmc.Card(
    children=[
        dmc.ChipGroup(
            [dmc.Chip(x['label'], value=x['value'], size='xs') for x in [
                {"label": "COSE", "value": "cose"},
                {"label": "Cent", "value": "concentric"},
                {"label": "Breadth", "value": "breadthfirst"},
                {"label": "Grid", "value": "grid"},
            ]],
            id="select_cyto_layout",
            value="cose",
        ),
        cyto.Cytoscape(
            id='cytoscape',
            elements=[],
            layout={'name': 'concentric'},
            style={'width': '100%', 'height': '100%'},
            stylesheet=[],
            responsive=True
        )

    ], withBorder=True,
    shadow="lg",
    radius="lg",
    style={'height': OVERVIEW_HEIGHT},
)

page_performance = html.Div(
    children=[
        dmc.Group(children=[
            dmc.Text("Client Performance", size=17, ml='md'),
            dmc.Select(id='select_client_metrics', size='sm', value="main", mb=0, ml='md')]
            , style={"height": 60}, align='center', mt='md'),
        dmc.Space(h='md'),
        dcc.Graph(id='figure_client_perform',
                  config={'displaylogo': False})
    ]
)
selection = html.Div(
    style={'width': OVERVIEW_WIDTH - 16},
    children=[dmc.Text('Select Clients', size=18),
              dmc.Space(h='md'),
              dmc.Grid(
                  [dmc.Col(dmc.TextInput(
                      id="client_selection_reg",
                      placeholder="Regex Filter",
                      style={'width': '100%'},
                      rightSection=DashIconify(icon="file-icons:regex"),
                  ), span='auto', mr=0),
                      dmc.Col(
                          dmc.ActionIcon(
                              DashIconify(icon='iconoir:list-select', width=24),
                              id="client_selection_check",
                              color="blue", variant="light", size=32, mr='sm'),
                          span='content')], mb='sm', align='center'),
              dmc.ChipGroup(
                  [],
                  id="client_selection_ms",
                  value=[],
                  mr='xs',
                  mb='lg',
                  multiple=True,
                  mah=100,
              )
              ])


def _gen_charts_grid(section, charts_config):
    grids = []
    for id, fig in charts_config[section].items():
        grids.append(dmc.Col(mb='sm', span=fig['span'],
                             children=dmc.LoadingOverlay(
                                 dcc.Graph(id=fig['name']
                                           , config={'autosizable': True, 'displaylogo': False}),
                                 loaderProps={"variant": "dots", "color": "blue", "size": "xl"})))
    return grids


def build_normal_charts(section, charts):
    grids = _gen_charts_grid(section, charts)
    return dmc.Grid(grids, mt='md')


def build_slider_charts(section, charts):
    global slider_index
    grids = _gen_charts_grid(section, charts)
    page_slider = html.Div(
        children=[
            dmc.Group(children=[
                dmc.Text("Communication Round", size=17, ml='md'),
                dmc.Slider(
                    id={'type': 'round_slider', 'section': section},
                    value=1,
                    updatemode="drag",
                    marks=[],
                    size='md',
                    ml='sm',
                    min=1,
                    max=1,
                    style={'width': '70%', 'height': 60}
                )],
                mt='md'
                , style={"height": 60, "width": "100%"}, align='center'),
            dmc.Grid(grids)]
    )
    return page_slider
