import unittest
import time
from copy import deepcopy

from fedlab.core.client import ORDINARY_TRAINER, SERIAL_TRAINER
from fedlab.core.network import DistNetwork
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.core.server.manager import ServerManager, SynchronousServerManager, AsynchronousServerManager
from fedlab.utils import MessageCode, Logger

from ..task_setting_for_test import CNN_Mnist

import torch
import torch.distributed as dist
from torch.multiprocessing import Process


