import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    
logging.getLogger().setLevel(logging.INFO)

class logger(object):
    """record cmd info to file and print it to cmd at the same time
    
    Args:
        log_file (str): a file path of log file
        log_name (str): log name for output
    """
    def __init__(self, log_file, log_name):
        self.logger = logging.getLogger(log_name)
        handler = logging.FileHandler(log_file, mode='w')
        handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, log_str):
        """Print information to logger"""
        self.logger.info(log_str)

    def warning(self, warning_str):
        """Print warning to logger"""
        self.logger.warning(warning_str)
