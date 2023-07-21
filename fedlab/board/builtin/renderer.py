import torch
from sklearn.manifold import TSNE

from fedlab.board import fedboard


def client_param_tsne(round: int, client_ids: list[str]):
    if len(client_ids) < 2:
        return None, None
    client_params = {}
    for role_id, params in fedboard.read_logged_obj(round, 'client_params').items():
        for k, v in params.items():
            client_params[k] = v
    raw_params = {str(id): param for id, param in client_params.items()}
    params_selected = [raw_params[id] for id in client_ids if id in raw_params.keys()]
    id_existed = [id for id in client_ids if id in raw_params.keys()]
    if len(params_selected) <= 1:
        return None, None
    params_selected = torch.stack(params_selected)
    params_tsne = TSNE(n_components=2, learning_rate=100, random_state=501,
                       perplexity=min(30.0, len(params_selected) - 1)).fit_transform(
        params_selected)
    return params_tsne, id_existed


def get_client_dataset_tsne(client_ids: list[str], type: str, size, client_ranks: list[str]):
    if len(client_ids) < 1:
        return None
    if not fedboard.get_delegate():
        return None
    raw = []
    client_range = {}
    for client_id, rank in zip(client_ids, client_ranks):
        data, label = fedboard.get_delegate().sample_client_data(client_id, rank, type, size)
        client_range[client_id] = (len(raw), len(raw) + len(data))
        raw += data
    if len(raw) == 0:
        return None
    raw = torch.stack(raw).view(len(raw), -1)
    tsne = TSNE(n_components=3, learning_rate=100, random_state=501,
                perplexity=min(30.0, len(raw) - 1)).fit_transform(raw)
    tsne = {cid: tsne[s:e] for cid, (s, e) in client_range.items()}
    return tsne


def get_client_data_report(clients_ids: list[str], type: str, client_ranks: list[str]):
    res = {}
    for client_id, rank in zip(clients_ids, client_ranks):
        def rd():
            if fedboard.get_delegate():
                return fedboard.get_delegate().read_client_label(client_id, rank, type=type)
            else:
                return {}

        obj = fedboard.read_obj_with_cache('data', 'partition', f'{client_id}', rd)
        res[client_id] = obj
    return res
