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

import os
import unittest
import torch
import torch.nn as nn
from fedlab.utils.serialization import SerializationTool


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SerializationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # DO NOT change the setting below, the model is pretrained on MNIST
        cls.input_size = 784
        cls.hidden_size = 250
        cls.num_classes = 10
        test_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(test_path, '../data/nnModel.ckpt')
        cls.model = Net(cls.input_size, cls.hidden_size, cls.num_classes)
        cls.model.load_state_dict(torch.load(model_path))

    def _model_params_eq(self, model1, model2):
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(param1, param2))

    def _model_params_neq(self, model1, model2):
        flags = []
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            flags.append(torch.equal(param1, param2))
        self.assertIn(False, flags)  # at least one False in flags

    def _model_params_multiply(self, model, times):
        for param in model.parameters():
            param.data = param.data * times

    def test_serialize_model_gradients(self):
        # set gradients with calculation
        batch_size = 100
        sample_data = torch.randn(batch_size, self.input_size)*255
        output = self.model(sample_data)
        res = output.mean()
        res.backward()
        # serialize gradients
        serialized_grads = SerializationTool.serialize_model_gradients(self.model)
        m_grads = torch.Tensor([0])
        for param in self.model.parameters():
            m_grads = torch.cat((m_grads, param.grad.data.view(-1)))
        m_grads = m_grads[1:]
        self.assertTrue(torch.equal(serialized_grads, m_grads))

    @torch.no_grad()
    def test_serialize_model(self):
        serialized_params = SerializationTool.serialize_model(self.model)
        m_params = torch.Tensor([0])
        for param in self.model.parameters():
            m_params = torch.cat((m_params, param.data.view(-1)))
        m_params = m_params[1:]
        self.assertTrue(torch.equal(serialized_params, m_params))

    @torch.no_grad()
    def test_deserialize_model_copy(self):
        model = Net(self.input_size, self.hidden_size, self.num_classes)
        self._model_params_neq(self.model, model)
        serialized_params = SerializationTool.serialize_model(self.model)
        SerializationTool.deserialize_model(model, serialized_params, mode='copy')
        self._model_params_eq(self.model, model)

    @torch.no_grad()
    def test_deserialize_model_add(self):
        model = Net(self.input_size, self.hidden_size, self.num_classes)
        self._model_params_neq(self.model, model)
        serialized_params = SerializationTool.serialize_model(self.model)
        SerializationTool.deserialize_model(model, serialized_params, mode='copy')  # copy first
        SerializationTool.deserialize_model(model, serialized_params, mode='add')  # add params then
        self._model_params_multiply(self.model, 2)  # now self.model.params = self.model.params * 2
        self._model_params_eq(self.model, model)

    @torch.no_grad()
    def test_deserialize_model_other(self):
        model = Net(self.input_size, self.hidden_size, self.num_classes)
        serialized_params = SerializationTool.serialize_model(self.model)
        with self.assertRaises(ValueError):
            SerializationTool.deserialize_model(model, serialized_params, mode='minus')

    

