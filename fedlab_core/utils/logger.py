import logging


class logger(object):
    """record cmd info to file and print it to cmd at the same time"""
    def __init__(self, log_file):
        self.logger = logging.getLogger(self.args.username)
        handler = logging.FileHandler(log_file, mode='w')
        handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, log_str):
        self.logger.info(log_str)

    def error(self, error_str, log_str):
        raise NotImplementedError()

    def warning(self, warning_str):
        raise NotImplementedError()