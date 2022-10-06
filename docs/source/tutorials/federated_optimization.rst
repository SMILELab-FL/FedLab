.. _federated-optimization:

**********************
Federated Optimization
**********************

Standard FL Optimization contains two parts: 1. local train in client; 2. global aggregation in server.  Local train and aggregation procedure are customizable in FedLab. You need to define :class:`ClientTrainer` and :class:`ServerHandler`.

Since :class:`ClientTrainer` and :class:`ServerHandler` are required to manipulate PyTorch Model. They are both inherited from :class:`ModelMaintainer`.

.. code-block:: python

    class ModelMaintainer(object):
        """Maintain PyTorch model.

        Provide necessary attributes and operation methods. More features with local or global model
        will be implemented here.

        Args:
            model (torch.nn.Module): PyTorch model.
            cuda (bool): Use GPUs or not.
            device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
        """
        def __init__(self,
                    model: torch.nn.Module,
                    cuda: bool,
                    device: str = None) -> None:
            self.cuda = cuda

            if cuda:
                # dynamic gpu acquire.
                if device is None:
                    self.device = get_best_gpu()
                else:
                    self.device = device
                self._model = deepcopy(model).cuda(self.device)
            else:
                self._model = deepcopy(model).cpu()

        def set_model(self, parameters: torch.Tensor):
            """Assign parameters to self._model."""
            SerializationTool.deserialize_model(self._model, parameters)

        @property
        def model(self) -> torch.nn.Module:
            """Return :class:`torch.nn.module`."""
            return self._model

        @property
        def model_parameters(self) -> torch.Tensor:
            """Return serialized model parameters."""
            return SerializationTool.serialize_model(self._model)

        @property
        def model_gradients(self) -> torch.Tensor:
            """Return serialized model gradients."""
            return SerializationTool.serialize_model_gradients(self._model)

        @property
        def shape_list(self) -> List[torch.Tensor]:
            """Return shape of model parameters.
            
            Currently, this attributes used in tensor compression.
            """
            shape_list = [param.shape for param in self._model.parameters()]
            return shape_list

Client local training
=======================

The basic class of ClientTrainer is shown below, we encourage users define local training process following our code pattern:

.. code-block:: python

    class ClientTrainer(ModelMaintainer):
        """An abstract class representing a client trainer.

        In FedLab, we define the backend of client trainer show manage its local model.
        It should have a function to update its model called :meth:`local_process`.

        If you use our framework to define the activities of client, please make sure that your self-defined class
        should subclass it. All subclasses should overwrite :meth:`local_process` and property ``uplink_package``.

        Args:
            model (torch.nn.Module): PyTorch model.
            cuda (bool): Use GPUs or not.
            device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to ``None``.
        """

        def __init__(self,
                    model: torch.nn.Module,
                    cuda: bool,
                    device: str = None) -> None:
            super().__init__(model, cuda, device)

            self.client_num = 1  # default is 1.
            self.dataset = FedDataset() # or Dataset
            self.type = ORDINARY_TRAINER

        def setup_dataset(self):
            """Set up local dataset ``self.dataset`` for clients."""
            raise NotImplementedError()

        def setup_optim(self):
            """Set up variables for optimization algorithms."""
            raise NotImplementedError()

        @property
        @abstractmethod
        def uplink_package(self) -> List[torch.Tensor]:
            """Return a tensor list for uploading to server.

                This attribute will be called by client manager.
                Customize it for new algorithms.
            """
            raise NotImplementedError()

        @abstractclassmethod
        def local_process(self, payload: List[torch.Tensor]):
            """Manager of the upper layer will call this function with accepted payload
            
                In synchronous mode, return True to end current FL round.
            """
            raise NotImplementedError()

        def train(self):
            """Override this method to define the training procedure. This function should manipulate :attr:`self._model`."""
            raise NotImplementedError()

        def validate(self):
            """Validate quality of local model."""
            raise NotImplementedError()

        def evaluate(self):
            """Evaluate quality of local model."""
            raise NotImplementedError()


- Overwrite :meth:`ClientTrainer.local_process()` to define local procedure. Typically, you need to implement standard training pipeline of PyTorch.
- Attributes ``model`` and ``model_parameters`` is is associated with ``self._model``. Please make sure the function ``local_process()`` will manipulate ``self._model``.

**A standard implementation of this part is in :class:`SGDClientTrainer`.**

Server global aggregation
==========================

Calculation tasks related with PyTorch should be define in ServerHandler part. In **FedLab**, our basic class of Handler is defined in :class:`ServerHandler`.

.. code-block:: python

    class ServerHandler(ModelMaintainer):
        """An abstract class representing handler of parameter server.

        Please make sure that your self-defined server handler class subclasses this class

        Example:
            Read source code of :class:`SyncServerHandler` and :class:`AsyncServerHandler`.
            
        Args:
            model (torch.nn.Module): PyTorch model.
            cuda (bool): Use GPUs or not.
            device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
        """
        def __init__(self,
                    model: torch.nn.Module,
                    cuda: bool,
                    device: str = None) -> None:
            super().__init__(model, cuda, device)

        @property
        @abstractmethod
        def downlink_package(self) -> List[torch.Tensor]:
            """Property for manager layer. Server manager will call this property when activates clients."""
            raise NotImplementedError()

        @property
        @abstractmethod
        def if_stop(self) -> bool:
            """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
            return False

        @abstractmethod
        def setup_optim(self):
            """Override this function to load your optimization hyperparameters."""
            raise NotImplementedError()

        @abstractmethod
        def global_update(self, buffer):
            raise NotImplementedError()

        @abstractmethod
        def load(self, payload):
            """Override this function to define how to update global model (aggregation or optimization)."""
            raise NotImplementedError()

        @abstractmethod
        def evaluate(self):
            """Override this function to define the evaluation of global model."""
            raise NotImplementedError()

User can define server aggregation strategy by finish following functions:

- You can overwrite ``_update_global_model()`` to customize global procedure.

- ``_update_global_model()`` is required to manipulate global model parameters (self._model).

- Summarised FL aggregation strategies are implemented in ``fedlab.utils.aggregator``.

**A standard implementation of this part is in SyncParameterServerHandler.**

