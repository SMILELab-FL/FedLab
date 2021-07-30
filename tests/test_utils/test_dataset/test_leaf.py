# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
import os
from fedlab_utils.dataset.leaf import dataloader


class FemnistTestCase(unittest.TestCase):
    """Tests for femnist dataset and dataloader module."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = '../../data/leaf'
        cls.dataset = 'femnist'
        # cls.client_id_map = {'f3797_07' : 1, 'f3793_06' : 2, 'f3728_28' : 3,
        #                      'f3687_48' : 4, 'f3785_26' : 5}
        cls.client_id_map = {1: 'f3797_07', 2: 'f3793_06', 3: 'f3728_28',
                             4: 'f3687_48', 5: 'f3785_26'}
        cls.train_num_samples = {'f3797_07': 11, 'f3793_06': 16, 'f3728_28': 17,
                                 'f3687_48': 18, 'f3785_26': 16}
        cls.test_num_samples = {'f3797_07': 19, 'f3793_06': 17, 'f3728_28': 19,
                                'f3687_48': 18, 'f3785_26': 16}

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_get_data(self) -> None:
        client_num, train_data_x_dict, train_data_y_dict, test_data_x_dict, test_data_y_dict \
            = dataloader.get_train_test_data(self.path)
        assert client_num == 5
        for client_id in range(client_num):
            client_value = self.client_id_map[client_id]
            assert train_data_x_dict[client_value] == self.train_num_samples[client_value]
            assert test_data_x_dict[client_value] == self.test_num_samples[client_value]

    def test_get_loader_for_femnist(self) -> None:
        dataloader.get_dataloader(dataset='femnist')
    # def test_load_model(self) -> None:
    #     """Test the number of (trainable) model parameters."""
    #     # pylint: disable=no-self-use
    #
    #     # Prepare
    #     expected = 62006
    #
    #     # Execute
    #     model: cifar.Net = cifar.load_model()
    #     actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    #     # Assert
    #     assert actual == expected
    #
    # def test_get_weights(self) -> None:
    #     """Test get_weights."""
    #     # pylint: disable=no-self-use
    #
    #     # Prepare
    #     model: cifar.Net = cifar.load_model()
    #     expected = 10
    #
    #     # Execute
    #     weights: Weights = model.get_weights()
    #
    #     # Assert
    #     assert len(weights) == expected
    #
    # def test_set_weights(self) -> None:
    #     """Test set_weights."""
    #     # pylint: disable=no-self-use
    #
    #     # Prepare
    #     weights_expected: Weights = cifar.load_model().get_weights()
    #     model: cifar.Net = cifar.load_model()
    #
    #     # Execute
    #     model.set_weights(weights_expected)
    #     weights_actual: Weights = model.get_weights()
    #
    #     # Assert
    #     for nda_expected, nda_actual in zip(weights_expected, weights_actual):
    #         np.testing.assert_array_equal(nda_expected, nda_actual)


if __name__ == "__main__":
    unittest.main()
