
from torch.multiprocessing import Process


class Topology(Process):
    """Abstract class

    Args:
        handler (`ClientBackendHandler` or `ParameterServerHandler`, optional): object to deal.
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
    """
    def __init__(self, network, handler=None):
        super(Topology, self).__init__()
        
        self._handler = handler
        self._network = network

    def run(self):
        raise NotImplementedError()

    def on_receive(self, sender, message_code, payload):
        raise NotImplementedError()
