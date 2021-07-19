from torch.multiprocessing import Process
from fedlab_core.network import DistNetwork


class Topology(Process):
    """Abstract class

    Args:
        handler (`ClientBackendHandler` or `ParameterServerHandler`, optional): object to deal.
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
    """
    def __init__(self, network: DistNetwork, handler=None):
        super(Topology, self).__init__()

        self._handler = handler
        self._network = network

    def run(self):
        raise NotImplementedError()

    def on_receive(self, sender, message_code, payload):
        raise NotImplementedError()
