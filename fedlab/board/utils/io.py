import configparser
import json
import os
import pickle
import shutil
from os import path
from typing import Any


def _clear_log(dir):
    shutil.rmtree(path.join(dir, f'log/'), ignore_errors=True)
    shutil.rmtree(path.join(dir, f'cache/'), ignore_errors=True)


def _update_meta_file(file_root: str, section: str, dct: dict):
    config_file = path.join(file_root, 'experiment.ini')
    os.makedirs(file_root, exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, 'w') as file:
            file.write('')
    config = configparser.ConfigParser()
    config.read(config_file)
    if not config.has_section(section):
        config.add_section(section)
    for key, value in dct.items():
        config.set(section, key, str(value))
    with open(config_file, 'w') as configfile:
        config.write(configfile)


def _read_meta_file(file_root: str, section: str, keys):
    config_file = path.join(file_root, 'experiment.ini')
    if not os.path.isfile(config_file):
        return None
    config = configparser.ConfigParser()
    config.read(config_file)
    if not config.has_section(section):
        return None
    res = {key: config.get(section, key) for key in keys}
    return res


def _log_to_fs(file_root: str, type: str, name: str, obj: Any, sub_type: str = None):
    pt = path.join(file_root, f'log/{type}/')
    if sub_type:
        pt = path.join(pt, f'{sub_type}/')
    os.makedirs(pt, exist_ok=True)
    pickle.dump(obj, open(path.join(pt, f'{name}.pkl'), 'wb+'))


def _log_to_fs_append(file_root: str, type: str, name: str, obj: Any, sub_type: str = None):
    pt = path.join(file_root, f'log/{type}/')
    if sub_type:
        pt = path.join(pt, f'{sub_type}/')
    os.makedirs(pt, exist_ok=True)
    with open(path.join(pt, f'{name}.log'), 'a+') as f:
        f.write(json.dumps(obj) + '\n')


def _read_log_from_fs(file_root: str, type: str, name: str, sub_type: str = None):
    target = path.join(file_root, f'log/{type}/')
    if sub_type:
        target = path.join(target, f'{sub_type}/')
    target = path.join(target, f'{name}.pkl')
    try:
        return pickle.load(open(target, 'rb'))
    except:
        return None


def _read_log_from_fs_appended(file_root: str, type: str, name: str, sub_type: str = None):
    target = path.join(file_root, f'log/{type}/')
    if sub_type:
        target = path.join(target, f'{sub_type}/')
    target = path.join(target, f'{name}.log')
    if not os.path.exists(target):
        return []
    return open(target).readlines()


def _read_cached_from_fs(file_root: str, type: str, sub_type: str, name: str):
    target = path.join(file_root, f'cache/{type}/{sub_type}/{name}.pkl')
    try:
        return pickle.load(open(target, 'rb'))
    except:
        return None


def _cache_to_fs(obj, file_root: str, type: str, sub_type: str, name: str):
    os.makedirs(path.join(file_root, f'cache/{type}/{sub_type}/'), exist_ok=True)
    target = path.join(file_root, f'cache/{type}/{sub_type}/{name}.pkl')
    pickle.dump(obj, open(target, 'wb+'))
