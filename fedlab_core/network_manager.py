from torch.multiprocessing import Process
from fedlab_core.network import DistNetwork


class NetworkManager(Process):
    """Abstract class

    Args:
        handler (`ClientBackendHandler` or `ParameterServerHandler`, optional): object to deal.
        newtork (`DistNetwork`): object to manage torch.distributed network communication.
    """
    def __init__(self, network: DistNetwork, handler=None):
        super(NetworkManager, self).__init__()

        self._handler = handler
        self._network = network

    def run(self):
        pass

    def on_receive(self, sender, message_code, payload):
        """Define the reaction of Topology get a package.
    
        Args:
            sender (int): rank of current process.
            message_code (`MessageCode`): message code
            payload (`torch.Tensor`): Tensor 
        """
        pass