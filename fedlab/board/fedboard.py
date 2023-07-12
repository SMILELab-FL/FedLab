import inspect
import json
import logging
import os
from os import path
from threading import Thread
from typing import Any

from dash import Dash

from fedlab.board.builtin.charts import add_built_in_charts
from fedlab.board.delegate import FedBoardDelegate
from fedlab.board.front.app import viewModel, create_app,_set_up_layout, _add_chart, _add_section,_add_callbacks
from fedlab.board.utils.io import _update_meta_file, _clear_log, _log_to_fs, _read_log_from_fs, _read_cached_from_fs, \
    _cache_to_fs, _log_to_fs_append, _read_log_from_fs_appended

_app: Dash | None = None
_delegate: FedBoardDelegate | None = None
_dir: str = ''


def get_delegate():
    return _delegate


def get_log_dir():
    return _dir


def setup(client_ids: list[str], max_round: int, name: str = None, log_dir: str = None):
    """
    Set up FedBoard
    Args:
        client_ids: List of client ids
        max_round: Max communication round
        name: Experiment name
        log_dir: Log directory
    """
    meta_info = {
        'name': name,
        'max_round': max_round,
        'client_ids': json.dumps(client_ids),
    }
    if log_dir is None:
        calling_file = inspect.stack()[1].filename
        calling_directory = os.path.dirname(os.path.abspath(calling_file))
        log_dir = calling_directory
    log_dir = path.join(log_dir, '.fedboard/')
    _update_meta_file(log_dir, 'meta', meta_info)
    _update_meta_file(log_dir, 'runtime', {'state': 'START', 'round': 0})
    global _app
    global _delegate
    global _dir
    _app = create_app(log_dir)
    _dir = log_dir
    _add_callbacks(_app)
    _clear_log(log_dir)


def enable_builtin_charts(delegate: FedBoardDelegate):
    """
    Enable builtin charts, including 'parameters' section and 'dataset' section.
    A dataset-reading delegate is required to enable these charts
    Args:
        delegate (FedBoardDelegate): dataset-reading delegate

    """
    global _delegate
    _delegate = delegate
    add_built_in_charts()


def start_offline(log_dir=None, port=8080):
    """
    Start Fedboard offline (seperated from the experiment)
    Args:
        log_dir: the experiment's log directory
        port: Which port will the board run in
    """
    if log_dir is None:
        calling_file = inspect.stack()[1].filename
        calling_directory = os.path.dirname(os.path.abspath(calling_file))
        log_dir = calling_directory
    log_dir = path.join(log_dir, '.fedboard/')
    add_built_in_charts()
    global _app
    _app = create_app(log_dir)
    _add_callbacks(_app)
    if _app is None:
        logging.error('FedBoard hasn\'t been initialized!')
        return
    _set_up_layout(_app)
    _app.run(host='0.0.0.0', port=port, debug=False, dev_tools_ui=True, use_reloader=False)


class RuntimeFedBoard():

    def __init__(self, port):
        meta_info = {
            'port': port,
        }
        _update_meta_file(_dir, 'meta', meta_info)
        self.port = port

    def _start_app(self):
        if _app is None:
            logging.error('FedBoard hasn\'t been initialized!')
            return
        _set_up_layout(_app)
        _app.run(host='0.0.0.0', port=self.port, debug=False, dev_tools_ui=True, use_reloader=False)

    def __enter__(self):
        self.p1 = Thread(target=self._start_app, daemon=True)
        self.p1.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.p1.join()


def log(round: int, metrics: dict[str, Any] = None, client_metrics: dict[str, dict[str, Any]] = None,
        main_metric_name: str = None, client_main_metric_name: str = None, **kwargs):
    """

    Args:

        round (int): Which communication round
        metrics (dict): Global performance at this round. E.g., {'loss':0.02, 'acc':0.85}
        client_metrics (dict): Client performance at this round. E.g., {'Client0':{'loss':0.01, 'acc':0.85}, 'Client1':...}
        main_metric_name (str): Main global metric. E.g., 'loss'
        client_main_metric_name (str): Main Client metric. E.g., 'acc'


    Returns:

    """
    state = "RUNNING"
    if round == viewModel.get_max_round():
        state = 'DONE'
    _update_meta_file(_dir, section='runtime', dct={'state': state, 'round': round})
    for key, obj in kwargs.items():
        _log_to_fs(_dir, type='params', sub_type=key, name=f'rd{round}', obj=obj)
    if metrics:
        if main_metric_name is None:
            main_metric_name = list[metrics.keys()][0]
        metrics['main_name'] = main_metric_name
        _log_to_fs_append(_dir, type='performs', name='overall', obj=metrics)
    if client_metrics:
        if client_main_metric_name is None:
            if main_metric_name:
                client_main_metric_name = main_metric_name
            else:
                client_main_metric_name = list(client_metrics[list(client_metrics.keys())[0]].keys())[0]
        for cid in client_metrics.keys():
            client_metrics[cid]['main_name'] = client_main_metric_name
        _log_to_fs_append(_dir, type='performs', name='client', obj=client_metrics)


def add_section(section: str, type: str):
    """

    Args:
        section (str): Section name
        type (str): Section type, can be 'normal' and 'slider', when set to 'slider', additional


    Returns:

    """
    assert type in ['normal', 'slider']
    _add_section(section=section, type=type)


def add_chart(section=None, figure_name=None, span=0.5):
    """
    Used as decorators for other functions,
    For sections with type = 'normal', the function takes input (selected_clients, selected_colors)
    For sections with type = 'slider', the function takes input (slider_value, selected_clients, selected_colors)

    Args:
        section (str): Section the chart will be added to
        figure_name (str) : Chart ID
        span (float): Chart span, E.g. 0.6 for 60% row width

    Examples:

        @add_chart('diy', 'slider', 1.0)
        def ct(slider_value, selected_clients, selected_colors):
            ...render the figure
            return figure

        @add_chart('diy2', 'slider', 1.0)
        def ct(slider_value, selected_clients, selected_colors):
            ...render the figure
            return figure

    """
    return _add_chart(section=section, figure_name=figure_name, span=span)


def read_logged_obj(round: int, type: str):
    return _read_log_from_fs(_dir, type='params', sub_type=type, name=f'rd{round}')


def read_logged_obj_appended(type: str, name: str, sub_type: str = None):
    return _read_log_from_fs_appended(_dir, type=type, name=name, sub_type=sub_type)


def read_cached_obj(type: str, sub_type: str, key: str, creator: callable):
    obj = _read_cached_from_fs(_dir, type=type, sub_type=sub_type, name=key)
    if not obj:
        obj = creator()
        _cache_to_fs(obj, _dir, type, sub_type, key)
    return obj
