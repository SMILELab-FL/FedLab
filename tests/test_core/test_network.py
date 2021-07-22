import unittest


from fedlab_core.network import DistNetwork



class NetworkTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.cnet = DistNetwork(address=("localhost","12345"), world_size=1, rank=0)
        self.cnet.init_network_connection()

    def tearDown(self) -> None:
        self.cnet.close_network_connection()
    

    def test_(self):
        self.cnet.show_configuration()