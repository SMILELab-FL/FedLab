import inspect
import logging
import os
import uuid
from os import path
from threading import Thread
from typing import Any

from dash import Dash

from fedlab.board import utils
from fedlab.board.builtin.charts import add_built_in_charts
from fedlab.board.delegate import FedBoardDelegate
from fedlab.board.front.app import viewModel, create_app, _set_up_layout, _add_chart, _add_section, _add_callbacks
from fedlab.board.utils.io import _update_meta_file, _log_to_fs, _read_log_from_fs, _read_cached_from_fs, \
    _cache_to_fs, _log_to_role_fs_append, get_role_ids, clear_log, clear_roles
from fedlab.board.utils.roles import *

_app: Dash | None = None
_initialized = False
_delegate: FedBoardDelegate | None = None
_dir: str = ''
_mode = 'standalone'
_role_id: str = ''
_roles: int = 3
_rank = 0


def register(id: str = None,
             log_dir: str = None,
             mode='standalone',
             roles: int = 3,
             process_rank: int = 0,
             client_ids: list[str] = None,
             max_round: int = None):
    """
    Register the process to FedBoard
    Args:
        id: Experiment id
        log_dir: Log directory
        mode(str): Should be 'standalone', or 'distributed'
            'standalone' for the experiment running on one machine, with multiple process perhaps
            'distributed' for the experiment running on multiple machines, in this case 'board_machine' must be specified
        process_rank: Process rank in the network
        client_ids: If the role of this process is 'Client holder', then register its client ids
        max_round: Max communication round

    """
    if not id:
        id = str(uuid.uuid4().hex)
    if log_dir is None:
        calling_file = inspect.stack()[1].filename
        calling_directory = os.path.dirname(os.path.abspath(calling_file))
        log_dir = calling_directory
    log_dir = path.join(log_dir, f'.fedboard/{id}/')
    global _dir, _mode, _rank, _role_id, _roles
    _mode = mode
    _dir = log_dir
    _rank = process_rank
    _roles = roles
    _role_id = f'{_roles}-{str(os.getpid())}-{process_rank}'
    if is_server(_roles):
        clear_log(log_dir)
        clear_roles(log_dir)
        _update_meta_file(_dir, 'runtime', {'state': 'START', 'round': 0})
    utils.io.register_role(_dir, _role_id)
    if client_ids:
        utils.io.register_client(_dir, _role_id, client_ids)
    if max_round:
        _update_meta_file(log_dir, 'meta', {'max_round': max_round})
    if is_board_shower(_roles):
        meta_info = {
            'name': id,
            'mode': mode,
        }
        _update_meta_file(log_dir, 'meta', meta_info)
        global _app
        _app = create_app(log_dir)
        _add_callbacks(_app)


def enable_builtin_charts(delegate: FedBoardDelegate):
    """
    Enable builtin charts, including 'parameters' section and 'dataset' section.
    A dataset-reading delegate is required to enable these charts
    To enable the 'client parameters' chart,
        all clients' parameters should be logged at every round, using a key = 'client_params'
    Args:
        delegate (FedBoardDelegate): dataset-reading delegate

    """
    if _dir is None:
        logging.error('FedBoard hasn\'t been initialized!')
        return
    if _mode != 'standalone':
        logging.error('Built-in charts are only supported in standalone mode')
        return
    global _delegate
    _delegate = delegate
    add_built_in_charts()


def start(port=8080):
    """
    Start Fedboard, the process will be blocked
    Args:
        port: Which port will the board run in
    """
    if _dir is None:
        logging.error('FedBoard hasn\'t been initialized!')
        return
    if not is_board_shower(_roles):
        logging.error('This process cannot start FedBoard!')
        return
    _set_up_layout(_app)
    _update_meta_file(_dir, 'meta', {'port': port})
    _app.run(host='0.0.0.0', port=port, debug=False, dev_tools_ui=True, use_reloader=False)


class RuntimeFedBoard:

    def __init__(self, port):
        self.port = port

    def __enter__(self):
        self.p1 = Thread(target=start, daemon=True, args=[self.port])
        self.p1.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.p1.join()


def log(round: int, metrics: dict[str, Any] = None, client_metrics: dict[str, dict[str, Any]] = None,
        main_metric_name: str = None, client_main_metric_name: str = None, **kwargs):
    """
    Log the intermediate results to FedBoard.

    Args:

        round (int): Which communication round
        metrics (dict): Global performance at this round. E.g., {'loss':0.02, 'acc':0.85}
        client_metrics (dict): Client performance at this round. E.g., {'Client0':{'loss':0.01, 'acc':0.85}, 'Client1':...}
        main_metric_name (str): Main global metric. E.g., 'loss'
        client_main_metric_name (str): Main Client metric. E.g., 'acc'

    """
    if _dir is None:
        logging.error('FedBoard hasn\'t been initialized!')
        return

    for key, obj in kwargs.items():
        _log_to_fs(_dir, _role_id, type='params', sub_type=key, name=f'rd{round}', obj=obj)

    if is_server(_roles):
        state = "RUNNING"
        if round == viewModel.get_max_round():
            state = 'DONE'
        _update_meta_file(_dir, section='runtime', dct={'state': state, 'round': round})
        if metrics:
            if main_metric_name is None:
                main_metric_name = next(iter(metrics))
            metrics['main_name'] = main_metric_name
            _log_to_role_fs_append(_dir, _role_id, section='performs', name='overall', round=round, obj=metrics)
    if is_client_holder(_roles) and client_metrics:
        if client_main_metric_name is None:
            if main_metric_name:
                client_main_metric_name = main_metric_name
            else:
                cd = client_metrics[next(iter(client_metrics))]
                client_main_metric_name = next(iter(cd))
        for cid in client_metrics.keys():
            client_metrics[cid]['main_name'] = client_main_metric_name
        _log_to_role_fs_append(_dir, _role_id, section='performs', name='client', round=round, obj=client_metrics)


def add_section(section: str, type: str):
    """
    Add a figure section to FedBoard

    Args:
        section (str): Section name
        type (str): Section type, can be 'normal' and 'slider', when set to 'slider', additional

    """
    if _dir is None:
        logging.error('FedBoard hasn\'t been initialized!')
        return
    if _mode != 'standalone':
        logging.error('Additional charts are only supported in standalone mode')
        return
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
    if _dir is None:
        logging.error('FedBoard hasn\'t been initialized!')
        return
    if _mode != 'standalone':
        logging.error('Additional charts are only supported in standalone mode')
        return
    return _add_chart(section=section, figure_name=figure_name, span=span)


def read_logged_obj(round: int, type: str) -> dict[str:Any]:
    f"""
    Read objects logged by all roles at a specific round
    Args:
        round: Global communication round
        type: Log type

    Returns:
        A dict whose keys are role_ids and values are logged objects
    """
    res = {}
    for role_id in get_all_roles():
        obj = _read_log_from_fs(_dir, role_id, type='params', sub_type=type, name=f'rd{round}')
        if obj:
            res[role_id] = obj
    return res


def read_logged_obj_current_process(round: int, type: str):
    return _read_log_from_fs(_dir, _role_id, type='params', sub_type=type, name=f'rd{round}')


def get_all_roles():
    return get_role_ids(_dir)


def get_delegate() -> FedBoardDelegate | None:
    """
    Get the dataset-reading delegate object
    Returns: delegate

    """
    return _delegate


def get_log_dir() -> str:
    """
    Get the logging directory
    Returns: dir

    """
    return _dir


def read_obj_with_cache(type: str, sub_type: str, key: str, creator: callable) -> Any:
    """
    Load from cache or create an object
    Args:
        type: object type
        sub_type: object sub-type
        key: object name
        creator: The object creating function

    Returns:
        object:Any, new created or loaded from cache

    """
    obj = _read_cached_from_fs(_dir, type=type, sub_type=sub_type, name=key)
    if not obj:
        obj = creator()
        _cache_to_fs(obj, _dir, type, sub_type, key)
    return obj
