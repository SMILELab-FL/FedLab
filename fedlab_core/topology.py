
from torch.multiprocessing import Process


class Topology(Process):
    """Abstract class

    Args:
        handler:
        newtork:
    """
    def __init__(self, handler, network):
        super(Topology, self).__init__()
        
        self._handler = handler
        self._network = network

    def run(self):
        raise NotImplementedError()

    def on_receive(self, sender, message_code, payload):
        raise NotImplementedError()
