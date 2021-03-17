import logging


class logger(object):
    """record cmd info to file"""

    def __init__(self, log_file):
        raise NotImplementedError()

    def log(self, log_str):
        raise NotImplementedError()
