import json
from os import path
from typing import Any

import diskcache
from dash import DiskcacheManager

from fedlab.board.utils.color import random_color
from fedlab.board.utils.data import encode_int_array
from fedlab.board.utils.io import _read_meta_file, _read_log_from_fs_appended


class ViewModel:

    def __init__(self):
        self.setup = False
        self.background_callback_manager = None

    def init(self, dir: str):
        self.dir = dir
        self.colors = {id: random_color(int(id)) for id in self.get_client_ids()}
        self.setup = True
        cache = diskcache.Cache(path.join(dir, "dash-cache/"))
        self.background_callback_manager = DiskcacheManager(cache)

    def get_color(self, client_id):
        return self.colors[client_id]

    def get_graph(self):
        elements = []
        elements.append(
            {"data": {"id": 'server', "label": 'Server', 'class_name': 'server'}})
        ss = [{'selector': 'node',
               'style': {
                   'label': 'data(label)'
               }}, {'selector': 'edge',
                    'style': {
                        "curve-style": "bezier",
                        'width': "2px",
                        'line-opacity': "50%"
                    }}, {'selector': f'[class_name *= "server"]',
                         'style': {
                             'background-color': '{}'.format('#5994FF'),
                             'shape': 'rectangle',
                         }}]

        for client_id in self.get_client_ids():
            elements.append(
                {"data": {"id": f'client{client_id}', "label": f'Clt{client_id}',
                          'class_name': f'client'}})
            elements.append({"data": {"source": 'server', "target": f'client{client_id}', "weight": 1,
                                      "dom_class_name": 'client'}})
            ss.append({'selector': f'[id *= "{client_id}"]',
                       'style': {
                           'background-color': f'{self.get_color(client_id)}',
                       }})

        return elements, ss

    def get_client_num(self):
        return len(self.get_client_ids())

    def get_client_ids(self):
        res = _read_meta_file(self.dir, "meta", ["client_ids"])['client_ids']
        res = json.loads(res)
        return res

    def client_id2index(self, client_id: str) -> int:
        res: str = _read_meta_file(self.dir, "meta", ["client_ids"])['client_ids']
        res: list[str] = json.loads(res)
        return res.index(client_id)

    def client_ids2indexes(self, client_ids: list[str]) -> list[int]:
        res: str = _read_meta_file(self.dir, "meta", ["client_ids"])['client_ids']
        res: list[str] = json.loads(res)
        return [res.index(id) for id in client_ids]

    def client_index2id(self, client_index: int) -> str:
        res = _read_meta_file(self.dir, "meta", ["client_ids"])['client_ids']
        res = json.loads(res)
        return res[client_index]

    def client_indexes2ids(self, client_indexes: list[int]) -> list[str]:
        res = _read_meta_file(self.dir, "meta", ["client_ids"])['client_ids']
        res = json.loads(res)
        return [res[idx] for idx in client_indexes]

    def get_max_round(self):
        res = _read_meta_file(self.dir, "meta", ["max_round"])
        return int(res['max_round'])

    def encode_client_ids(self, client_ids: list[str]):
        client_indexes = self.client_ids2indexes(client_ids)
        return encode_int_array(client_indexes)

    def get_overall_metrics(self):
        main_name = ""
        metrics = []
        log_lines = _read_log_from_fs_appended(self.dir, type='performs', name='overall')
        if len(log_lines) > 1:
            obj: dict[str, Any] = json.loads(log_lines[-1])
            main_name = obj['main_name']
            metrics = [k for k in obj.keys() if k != 'main_name']
        return metrics, main_name

    def get_client_metrics(self):
        main_name = ""
        metrics = []
        log_lines = _read_log_from_fs_appended(self.dir, type='performs', name='client')
        if len(log_lines) > 1:
            obj: dict[str, dict[str:Any]] = json.loads(log_lines[-1])
            if len(obj.keys()) > 0:
                clt_dct = obj[list(obj.keys())[0]]
                main_name = clt_dct['main_name']
                metrics = [k for k in clt_dct.keys() if k != 'main_name']
        return metrics, main_name

    def get_overall_performance(self):
        res_all = []
        main_name = ""
        log_lines = _read_log_from_fs_appended(self.dir, type='performs', name='overall')
        for line in log_lines:
            obj = json.loads(line)
            main_name = obj['main_name']
            res_all.append(obj)
        return res_all, main_name

    def get_client_performance(self, client_ids: list[str]):
        res = {}
        main_name = ""
        log_lines = _read_log_from_fs_appended(self.dir, type='performs', name='client')
        for line in log_lines:
            obj = json.loads(line)
            for client_id in client_ids:
                main_name = obj[client_id]['main_name']
                res.setdefault(client_id, [])
                res[client_id].append(obj[client_id])
        return res, main_name
