import json
import os.path
import pickle
from os import path
from typing import Any

import diskcache
import torch
from dash import DiskcacheManager
from sklearn.manifold import TSNE

from fedlab.board.delegate import FedBoardDelegate
from fedlab.board.utils.color import random_color
from fedlab.board.utils.data import encode_int_array
from fedlab.board.utils.io import _read_meta_file


class ViewModel:

    def __init__(self):
        self.setup = False
        self.background_callback_manager = None

    def init(self, dir: str, delegate: FedBoardDelegate = None):
        self.delegate = delegate
        self.dir = dir
        self.colors = {id: random_color(int(id)) for id in self.get_client_ids()}
        self.setup = True
        cache = diskcache.Cache(path.join(dir, "cache/"))
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
        res = _read_meta_file(self.dir, "meta", ["client_ids"])['client_ids']
        res: list[str] = json.loads(res)
        return res.index(client_id)

    def client_ids2indexes(self, client_ids: list[str]) -> list[int]:
        res = _read_meta_file(self.dir, "meta", ["client_ids"])['client_ids']
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

    def client_param_tsne(self, round: int, client_ids: list[str]):
        if not os.path.exists(os.path.join(self.dir, f'params/raw/rd{round}.pkl')):
            return None
        if len(client_ids) < 2:
            return None
        raw_params = {str(id): param for id, param in
                      pickle.load(open(os.path.join(self.dir, f'params/raw/rd{round}.pkl'), 'rb')).items()}
        # fn = self.encode_client_ids(client_ids)
        # target_file = os.path.join(self.dir, f'params/tsne/rd{round}/{fn}.pkl')
        # if os.path.exists(target_file):
        #     return pickle.load(open(target_file, 'rb'))
        os.makedirs(os.path.join(self.dir, f'params/tsne/rd{round}/'), exist_ok=True)
        params_selected = [raw_params[id][0] for id in client_ids if id in raw_params.keys()]
        if len(params_selected) < 1:
            return None
        params_selected = torch.stack(params_selected)
        params_tsne = TSNE(n_components=2, learning_rate=100, random_state=501,
                           perplexity=min(30.0, len(params_selected) - 1)).fit_transform(
            params_selected)
        # pickle.dump(params_tsne, open(target_file, 'wb+'))
        return params_tsne

    def get_client_dataset_tsne(self, client_ids: list, type: str, size):
        if len(client_ids) < 2:
            return None
        if not self.delegate:
            return None
        # client_indexes = self.client_ids2indexes(client_ids)
        # fn = encode_int_array(client_indexes)
        # target_file = os.path.join(self.dir, f'data/tsne/{fn}.pkl')
        # if os.path.exists(target_file):
        #     return pickle.load(open(target_file, 'rb'))
        os.makedirs(os.path.join(self.dir, f'data/tsne/'), exist_ok=True)
        raw = []
        client_range = {}
        for client_id in client_ids:
            data, label = self.delegate.sample_client_data(client_id, type, size)
            client_range[client_id] = (len(raw), len(raw) + len(data))
            raw += data
        raw = torch.stack(raw).view(len(raw), -1)
        tsne = TSNE(n_components=3, learning_rate=100, random_state=501,
                    perplexity=min(30.0, len(raw) - 1)).fit_transform(raw)
        tsne = {cid: tsne[s:e] for cid, (s, e) in client_range.items()}
        # pickle.dump(tsne, open(target_file, 'wb+'))
        return tsne

    def get_client_data_report(self, clients_ids: list, type: str):
        res = {}
        for client_id in clients_ids:
            target_file = os.path.join(self.dir, f'data/partition/{client_id}.pkl')
            if os.path.exists(target_file):
                res[client_id] = pickle.load(open(target_file, 'rb'))
            else:
                os.makedirs(os.path.join(self.dir, f'data/partition/'), exist_ok=True)
                if self.delegate:
                    res[client_id] = self.delegate.read_client_label(client_id, type=type)
                else:
                    res[client_id] = {}
                pickle.dump(res[client_id], open(target_file, 'wb+'))
        return res

    def get_overall_metrics(self):
        main_name = ""
        metrics = []
        if not os.path.exists(path.join(self.dir, f'performs/overall')):
            return metrics, main_name
        log_lines = open(path.join(self.dir, f'performs/overall')).readlines()
        if len(log_lines) > 1:
            obj: dict[str, Any] = json.loads(log_lines[-1])
            main_name = obj['main_name']
            metrics = [k for k in obj.keys() if k != 'main_name']
        return metrics, main_name

    def get_client_metrics(self):
        main_name = ""
        metrics = []
        if not os.path.exists(path.join(self.dir, f'performs/client')):
            return metrics, main_name
        log_lines = open(path.join(self.dir, f'performs/client')).readlines()
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
        if not os.path.exists(path.join(self.dir, f'performs/overall')):
            return res_all, main_name
        for line in open(path.join(self.dir, f'performs/overall')).readlines():
            obj = json.loads(line)
            main_name = obj['main_name']
            res_all.append(obj)
        return res_all, main_name

    def get_client_performance(self, client_ids: list[str]):
        res = {}
        main_name = ""
        if not os.path.exists(path.join(self.dir, f'performs/client')):
            return res, main_name
        for line in open(path.join(self.dir, f'performs/client')).readlines():
            obj = json.loads(line)
            for client_id in client_ids:
                main_name = obj[client_id]['main_name']
                res.setdefault(client_id, [])
                res[client_id].append(obj[client_id])
        return res, main_name
