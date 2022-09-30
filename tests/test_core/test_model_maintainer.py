import unittest
import os
from copy import deepcopy

import torch
import torch.nn as nn
from fedlab.core.model_maintainer import ModelMaintainer, SerialModelMaintainer
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


def _make_trained_model_and_params():
    # get trained model and serialized params loaded from local file
    input_size = 784
    hidden_size = 250
    num_classes = 10
    model = Net(input_size, hidden_size, num_classes)
    test_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(test_path, '../data/nnModel.ckpt')
    model.load_state_dict(torch.load(model_path))
    serialized_params = SerializationTool.serialize_model(model)
    return model, serialized_params


class ModelMaintainerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_size = 784
        cls.hidden_size = 250
        cls.num_classes = 10
        
    def setUp(self):
        self.model = Net(self.input_size, self.hidden_size, self.num_classes)

    def _model_params_eq(self, model1, model2):
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(param1, param2))

    def _model_params_neq(self, model1, model2):
        flags = []
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            flags.append(torch.equal(param1, param2))
        self.assertIn(False, flags)  # at least one False in flags

    def test_init(self):
        self._init_cuda_False()
        self._init_cuda_True_device_None()
        self._init_cuda_True_device_not_None()

    def _init_cuda_False(self):
        maintainer = ModelMaintainer(self.model, cuda=False)
        # check model should be on 'cpu'
        device = str(next(maintainer._model.parameters()).device)
        self.assertEqual(device, 'cpu')
        self.assertFalse(maintainer.cuda)

    def _init_cuda_True_device_None(self):
        # set cuda=True, device=None
        maintainer = ModelMaintainer(self.model, cuda=True)
        # check device property
        real_device = str(next(maintainer._model.parameters()).device)
        self.assertTrue('cuda' in real_device)
        self.assertTrue('cuda' in str(maintainer.device))
        # check cuda property
        self.assertTrue(maintainer.cuda)

    def _init_cuda_True_device_not_None(self):
        # set cuda=True, device='cuda:0'
        device = 'cuda:0'
        maintainer = ModelMaintainer(self.model, cuda=True, device=device)
        # check device property
        real_device = str(next(maintainer._model.parameters()).device)
        self.assertEqual(real_device, device)
        self.assertEqual(str(maintainer.device), device)
        # check cuda property
        self.assertTrue(maintainer.cuda)

    @torch.no_grad()
    def test_set_model(self):
        # get trained model and params loaded from local file
        new_model, serialized_new_params = _make_trained_model_and_params()
        # init model maintainer
        maintainer = ModelMaintainer(self.model, cuda=False)  # set cuda=False for comparison convenience
        self._model_params_neq(maintainer._model, new_model)  # should not be equal after initialization
        maintainer.set_model(serialized_new_params)  # set model params now
        self._model_params_eq(maintainer._model, new_model)  # should be equal after set_model

    def test_model_parameters(self):
        maintainer = ModelMaintainer(self.model, cuda=False)
        serialized_params1 = maintainer.model_parameters
        serialized_params2 = SerializationTool.serialize_model(self.model)
        self.assertTrue(torch.equal(serialized_params1, serialized_params2))

    def test_model_gradients(self):
        maintainer = ModelMaintainer(self.model, cuda=False)
        # set gradients with calculation
        batch_size = 100
        sample_data = torch.randn(batch_size, self.input_size)*255
        output = maintainer._model(sample_data)
        res = output.mean()
        res.backward()
        # serialize gradients
        serialized_grads = maintainer.model_gradients
        m_grads = torch.Tensor([0])
        for param in maintainer._model.parameters():
            m_grads = torch.cat((m_grads, param.grad.data.view(-1)))
        m_grads = m_grads[1:]
        self.assertTrue(torch.equal(serialized_grads, m_grads))

    def test_shape_list(self):
        self._check_shape_list(cuda=False)  # check shape list when model on cpu
        self._check_shape_list(cuda=True)  # check shape list when model on cuda

    def _get_shape_list(self, model):
        return [param.shape for param in model.parameters()]

    def _shape_list_eq(self, shapes1, shapes2):
        self.assertEqual(len(shapes1), len(shapes2))
        for idx in range(len(shapes1)):
            self.assertTrue(shapes1[idx] == shapes2[idx])

    def _check_shape_list(self, cuda):
        maintainer = ModelMaintainer(self.model, cuda=cuda)
        shape_list1 = self._get_shape_list(self.model)
        shape_list2 = maintainer.shape_list
        self._shape_list_eq(shape_list1, shape_list2)


class SerialModelMaintainerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_size = 784
        cls.hidden_size = 250
        cls.num_classes = 10
        cls.num_clients = 10
        
    def setUp(self):
        self.model = Net(self.input_size, self.hidden_size, self.num_classes)

    def test_init(self):
        self._init_personal_False()
        self._init_personal_True()

    def _model_params_eq(self, model1, model2):
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(param1, param2))

    def _init_personal_False(self):
        maintainer = SerialModelMaintainer(self.model, num_clients=self.num_clients, cuda=False, personal=False)
        self.assertTrue(maintainer.parameters is None)

    def _init_personal_True(self):
        maintainer = SerialModelMaintainer(self.model, num_clients=self.num_clients, cuda=False, personal=True)
        self.assertEqual(len(maintainer.parameters), self.num_clients)

    def test_set_model(self):
        self._set_model_id_None()
        self._set_model_id_not_None()

    def _set_model_id_None(self):
        # load prepared model params
        new_model, serialized_new_params = _make_trained_model_and_params()
        # init model maintainer
        maintainer = SerialModelMaintainer(self.model, num_clients=self.num_clients, cuda=False, personal=False)
        maintainer.set_model(parameters=serialized_new_params, id=None)
        self._model_params_eq(maintainer._model, new_model)  # should be equal after set_model

    def _set_model_id_not_None(self):
        # load prepared model params
        new_model, serialized_new_params = _make_trained_model_and_params()
        # init model maintainer
        maintainer = SerialModelMaintainer(self.model, num_clients=self.num_clients, cuda=False, personal=True)
        maintainer.parameters[5] = deepcopy(serialized_new_params)  # change model parameters of client_id=5 directly
         # set maintainer._model's parameter as client_id=5
        maintainer.set_model(id=5) 
        self._model_params_eq(maintainer._model, new_model)



    