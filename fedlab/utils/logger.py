# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.getLogger().setLevel(logging.INFO)


class Logger(object):
    """record cmd info to file and print it to cmd at the same time
    
    Args:
        log_name (str): log name for output.
        log_file (str): a file path of log file.
    """
    def __init__(self, log_name, log_file=None):
        self.logger = logging.getLogger(log_name)

        if log_file is not None:
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
