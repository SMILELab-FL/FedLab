import json
from os import path
from typing import Any

import diskcache
from dash import DiskcacheManager

from fedlab.board.utils.color import random_color
from fedlab.board.utils.data import encode_int_array
from fedlab.board.utils.io import _read_meta_file, get_client_ids, \
    _read_log_from_role_fs_appended, get_server_role_ids, get_roles_tree


class ViewModel:

    def __init__(self):
        self.setup = False
        self.background_callback_manager = None

    def init(self, dir: str):
        self.dir = dir
        self.colors = {}
        self.setup = True
        cache = diskcache.Cache(path.join(dir, "dash-cache/"))
        self.background_callback_manager = DiskcacheManager(cache)

    def get_color(self, client_id):
        if client_id not in self.colors.keys():
            self.colors[client_id] = random_color(self.client_id2index(client_id))
        return self.colors[client_id]

    def get_graph(self):
        elements = []

        ss = [{'selector': 'node',
               'style': {
                   'label': 'data(label)'
               }},
              {'selector': 'edge',
               'style': {
                   "curve-style": "bezier",
                   'width': "2px",
                   'line-opacity': "50%"
               }},
              {'selector': f'[class_name *= "server"]',
               'style': {
                   'background-color': '{}'.format('#5994FF'),
                   'shape': 'rectangle',
                   'width': 45,
                   'height': 45,
               }},
              {'selector': f'[class_name *= "process"]',
               'style': {
                   'background-color': '{}'.format('#000000'),
                   'shape': 'triangle',
                   'width': 40,
                   'height': 40,
               }},
              {'selector': f'[dom_class_name *= "cs"]',
               'style': {
                   'target-arrow-shape': 'triangle',
                   'target-arrow-color': '#505050',
                   'line-color': '{}'.format('#808080'),
                   'line-opacity': "30%",
                   'width': "5px",
               }},
              ]

        roles: dict[str, list[dict]] = get_roles_tree(self.dir)
        pure_holder_ids = []
        server_holder_ids = []
        for role_id, children in roles.items():
            elements.append(
                {"data": {"id": f'process-{role_id}', "label": f'Process{role_id.split("-")[-1]}',
                          'class_name': 'process'}})
            pure_client = True
            for child in children:
                if child['role'] == 'server':
                    elements.append(
                        {"data": {"id": f'server-{role_id}', "label": 'Server', 'class_name': 'server'}})
                    elements.append(
                        {"data": {"source": f'process-{role_id}', "target": f'server-{role_id}', "weight": 0.5,
                                  "dom_class_name": 'cp'}})
                    pure_client = False

            if pure_client:
                pure_holder_ids.append(role_id)
            else:
                server_holder_ids.append(role_id)
            for child in children:
                if 'client_ids' in child.keys():
                    for client_id in child['client_ids']:
                        elements.append(
                            {"data": {"id": f'client-{client_id}', "label": f'Clt{client_id}',
                                      'class_name': f'client'}})
                        ss.append({'selector': f'[id *= "client-{client_id}"]',
                                   'style': {
                                       'background-color': f'{self.get_color(client_id)}',
                                   }})
                        if pure_client:
                            elements.append(
                                {"data": {"source": f'process-{role_id}', "target": f'client-{client_id}', "weight": 1,
                                          "dom_class_name": 'cp'}})
                        else:
                            elements.append(
                                {"data": {"source": f'server-{role_id}', "target": f'client-{client_id}', "weight": 1,
                                          "dom_class_name": 'cs'}})
        for pure_holder_id in pure_holder_ids:
            elements.append(
                {"data": {"source": f'process-{server_holder_ids[-1]}', "target": f'process-{pure_holder_id}',
                          "weight": 2,
                          "dom_class_name": 'cs'}})
        return elements, ss

    def get_client_num(self):
        return len(self.get_client_ids())

    def get_client_ids(self) -> list[str]:
        id_dict = get_client_ids(self.dir)
        # res = _read_meta_file(self.dir, "meta", ["client_ids"])['client_ids']
        # res = json.loads(res)
        res = []
        for k, v in id_dict.items():
            res += v
        return res

    def get_client_holders(self) -> dict[str]:
        id_dict = get_client_ids(self.dir)
        return id_dict

    def client_ids2ranks(self, client_ids: list[str]):
        id_dict = get_client_ids(self.dir)
        mapper = {}
        for k, v in id_dict.items():
            for id in v:
                mapper[id] = k.split('-')[-1]
        return [mapper[id] for id in client_ids]

    def client_id2index(self, client_id: str) -> int:
        res = self.get_client_ids()
        return res.index(client_id)

    def client_ids2indexes(self, client_ids: list[str]) -> list[int]:
        res = self.get_client_ids()
        return [res.index(id) for id in client_ids]

    def client_index2id(self, client_index: int) -> str:
        res = self.get_client_ids()
        return res[client_index]

    def client_indexes2ids(self, client_indexes: list[int]) -> list[str]:
        res = self.get_client_ids()
        return [res[idx] for idx in client_indexes]

    def get_max_round(self):
        try:
            res = _read_meta_file(self.dir, "meta", ["max_round"])
            return int(res['max_round'])
        except:
            return -1

    def encode_client_ids(self, client_ids: list[str]):
        client_indexes = self.client_ids2indexes(client_ids)
        return encode_int_array(client_indexes)

    def get_overall_metrics(self):
        main_name = ""
        metrics = set()
        for role_id in get_server_role_ids(self.dir):
            log_lines = _read_log_from_role_fs_appended(self.dir, role_id, section='performs', name='overall')
            if len(log_lines) > 1:
                obj: dict[str, Any] = json.loads(log_lines[-1].split('==')[-1])
                main_name = obj['main_name']
                metrics = metrics.union(set(k for k in obj.keys() if k != 'main_name'))
        return list(metrics), main_name

    def get_overall_performance(self):
        res_all = []
        main_name = ""
        for role_id in get_server_role_ids(self.dir):
            log_lines = _read_log_from_role_fs_appended(self.dir, role_id, section='performs', name='overall')
            for line in log_lines:
                round = int(line.split('==')[0])
                obj = json.loads(line.split('==')[1])
                main_name = obj['main_name']
                res_all.append((round, obj))
            break
        return res_all, main_name

    def get_client_performance(self, client_ids: list[str]):
        res = {}
        main_name = ""
        chs = self.get_client_holders().keys()
        for role_id in chs:
            log_lines = _read_log_from_role_fs_appended(self.dir, role_id, section='performs', name='client')
            for line in log_lines:
                round = int(line.split("==")[0])
                obj: dict[str, dict] = json.loads(line.split("==")[1])
                for client_id in client_ids:
                    if client_id in obj.keys():
                        main_name = obj[client_id]['main_name']
                        res.setdefault(client_id, [])
                        res[client_id].append((round, obj[client_id]))
        return res, main_name

    def get_client_metrics(self):
        main_name = ""
        metrics = set()
        chs = self.get_client_holders().keys()
        for role_id in chs:
            log_lines = _read_log_from_role_fs_appended(self.dir, role_id, section='performs', name='client')
            if len(log_lines) > 1:
                obj: dict[str, dict[str:Any]] = json.loads(log_lines[-1].split('==')[-1])
                if len(obj.keys()) > 0:
                    clt_dct = obj[list(obj.keys())[0]]
                    main_name = clt_dct['main_name']
                    metrics = metrics.union(set(k for k in clt_dct.keys() if k != 'main_name'))

        return list(metrics), main_name
