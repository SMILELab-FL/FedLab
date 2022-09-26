
from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer


##################
#
#      Server
#
##################


class FedAvgServerHandler(SyncServerHandler):
    """FedAvg server handler."""
    None


##################
#
#      Client
#
##################


class FedAvgClientTrainer(SGDClientTrainer):
    """Federated client with local SGD solver."""
    None