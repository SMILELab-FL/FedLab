import unittest
import torch
from fedlab_benchmarks.datasets.leaf_data_process.dataloader import get_train_test_data, get_dataloader
from fedlab_benchmarks.datasets.leaf_data_process.femnist import process as process_femnist


class LeafTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.datapath = '../../../tests/data/leaf'
        self.dataset = 'femnist'
        self.client_id_name_map = {4: 'f3797_07', 3: 'f3793_06', 1: 'f3728_28', 0: 'f3687_48', 2: 'f3785_26'}
        self.train_sample_num_map = {'f3797_07': 11, 'f3793_06': 16, 'f3728_28': 17, 'f3687_48': 18, 'f3785_26': 16}
        self.test_sample_num_map = {'f3797_07': 19, 'f3793_06': 17, 'f3728_28': 19, 'f3687_48': 18, 'f3785_26': 16}

    def tearDown(self) -> None:
        return super().tearDown()

    def test_get_data_client(self):
        for client_id in range(5):
            train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_data(self.datapath,
                                                                                       client_id)
            client_name = self.client_id_name_map[client_id]
            assert len(train_data_x) == len(train_data_y)
            assert len(train_data_x) == self.train_sample_num_map[client_name]
            assert len(test_data_x) == len(test_data_y)
            assert len(test_data_x) == self.test_sample_num_map[client_name]

    def test_data_process_leaf(self):
        train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_data(self.datapath,
                                                                                   client_id=0,
                                                                                   process_x=process_femnist.process_x,
                                                                                   process_y=process_femnist.process_y)
        assert train_data_x.shape[1:] == (1, 28, 28)
        assert isinstance(train_data_y[0], torch.LongTensor)


if __name__ == '__main__':
    unittest.main()
