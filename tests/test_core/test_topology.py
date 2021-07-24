import unittest


from fedlab_core.network_manager import NetworkManager



class TopologyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_topology(self):
        top = NetworkManager(network=None)
        top.on_receive(sender=None, message_code=None, payload=None)
        