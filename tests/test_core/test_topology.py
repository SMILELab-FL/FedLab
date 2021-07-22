import unittest


from fedlab_core.topology import Topology



class TopologyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_topology(self):
        top = Topology(network=None)
        top.on_receive(sender=None, message_code=None, payload=None)
        