import configparser
import os
import shutil
from os import path


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
