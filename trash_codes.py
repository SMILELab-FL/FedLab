

class ServerBasicTopology(Process, ABC):
    """Abstract class for server network topology

    If you want to define your own topology agreements, please subclass it.

    Args:
        server_address (tuple): Address of server in form of ``(SERVER_ADDR, SERVER_IP)``
        dist_backend (str): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``
    """
    def __init__(self, handler, network):
        self._handler = handler
        self._network = network

    @abstractmethod
    def run(self):
        """Main process, define your server's behavior"""
        raise NotImplementedError()

    def on_receive(self, sender, message_code, payload):
        raise NotImplementedError()

    def shutdown_clients(self):
        """Shutdown all clients"""
        for client_idx in range(1, self._handler.client_num_in_total + 1):
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=client_idx)


class ClientBasicTopology(Process, ABC):
    """Abstract class of client topology

    If you want to define your own Network Topology, please be sure your class should subclass it and OVERRIDE its methods.

    Example:
        Read the code of :class:`ClientPassiveTopology` and `ClientActiveTopology` to learn how to use this class.
    """
    def __init__(self, handler, network):
        self._handler = handler
        self._network = network

    @abstractmethod
    def run(self):
        """Please override this function"""
        raise NotImplementedError()

    @abstractmethod
    def on_receive(self, sender_rank, message_code, payload):
        """Please override this function"""
        raise NotImplementedError()

    @abstractmethod
    def synchronize(self):
        """Please override this function"""
        raise NotImplementedError()