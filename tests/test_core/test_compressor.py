import unittest

from fedlab_utils.models.lenet import LeNet
from fedlab_core.communicator.compressor import TopkCompressor


class CompressorTestCase(unittest.TestCase):
    def setUp(self) -> None:

        self.model = LeNet()
        self.compressor = TopkCompressor(compress_ratio=0.5)
        self.compressor.initialize(self.model.named_parameters())

        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_compress(self):
        for name, param in self.model.named_parameters():
            tensor_info, ctx = self.compressor.compress(param, name)
            decompressed = self.compressor.decompress(tensor_info, ctx)

            assert decompressed.shape == param.shape
