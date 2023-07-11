import inspect
import json
import logging
import os
import pickle
from os import path
from threading import Thread
from typing import Any

from dash import Dash

from fedlab.board.builtin.charts import _add_built_in_charts
from fedlab.board.delegate import FedBoardDelegate
from fedlab.board.front.app import viewModel, create_app, add_callbacks, set_up_layout, _add_chart, _add_section
from fedlab.board.utils.io import _update_meta_file, _clear_log

_app: Dash | None = None


def setup(delegate: FedBoardDelegate, client_ids, max_round, name=None, log_dir=None):
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
    _add_built_in_charts()
    _app = create_app(log_dir, delegate)
    add_callbacks(_app)
    _clear_log(log_dir)


def start_offline(log_dir=None, port=8080):
    if log_dir is None:
        calling_file = inspect.stack()[1].filename
        calling_directory = os.path.dirname(os.path.abspath(calling_file))
        log_dir = calling_directory
    log_dir = path.join(log_dir, '.fedboard/')
    _add_built_in_charts()
    global _app
    _app = create_app(log_dir)
    add_callbacks(_app)
    if _app is None:
        logging.error('FedBoard hasn\'t been initialized!')
        return
    set_up_layout(_app)
    _app.run(host='0.0.0.0', port=port, debug=False, dev_tools_ui=True, use_reloader=False)


class RuntimeFedBoard():

    def __init__(self, port):
        meta_info = {
            'port': port,
        }
        _update_meta_file(viewModel.dir, 'meta', meta_info)
        self.port = port

    def _start_app(self):
        if _app is None:
            logging.error('FedBoard hasn\'t been initialized!')
            return
        set_up_layout(_app)
        _app.run(host='0.0.0.0', port=self.port, debug=False, dev_tools_ui=True, use_reloader=False)

    def __enter__(self):
        self.p1 = Thread(target=self._start_app, daemon=True)
        self.p1.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.p1.join()


def log(round: int, client_params: dict[str, Any] = None, metrics: dict[str, Any] = None,
        main_metric_name: str = None, client_metrics: dict[str, dict[str, Any]] = None):
    state = "RUNNING"
    if round == viewModel.get_max_round():
        state = 'DONE'
    _update_meta_file(viewModel.dir, section='runtime', dct={'state': state, 'round': round})
    if client_params:
        os.makedirs(path.join(viewModel.dir, f'log/params/raw/'), exist_ok=True)
        pickle.dump(client_params, open(path.join(viewModel.dir, f'log/params/raw/rd{round}.pkl'), 'wb+'))
    if metrics:
        os.makedirs(path.join(viewModel.dir, f'log/performs/'), exist_ok=True)
        if main_metric_name is None:
            main_metric_name = list[metrics.keys()][0]
        metrics['main_name'] = main_metric_name
        with open(path.join(viewModel.dir, f'log/performs/overall'), 'a+') as f:
            f.write(json.dumps(metrics) + '\n')
    if client_metrics:
        os.makedirs(path.join(viewModel.dir, f'log/performs/'), exist_ok=True)
        if main_metric_name is None:
            main_metric_name = list(client_metrics[list(client_metrics.keys())[0]].keys())[0]
        for cid in client_metrics.keys():
            client_metrics[cid]['main_name'] = main_metric_name
        with open(path.join(viewModel.dir, f'log/performs/client'), 'a+') as f:
            f.write(json.dumps(client_metrics) + '\n')


def add_section(section: str, type: str):
    _add_section(section=section, type=type)


def add_chart(section=None, figure_name=None, span=6):
    return _add_chart(section=section, figure_name=figure_name, span=span)
