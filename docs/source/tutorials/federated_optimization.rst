.. _federated-optimization:

**********************
Federated Optimization
**********************

Standard FL Optimization contains two parts: 1. local train in client; 2. global aggregation in server.  Local train and aggregation procedure are customizable in FedLab. You need to define :class:`ClientTrainer` and :class:`ParameterServerBackendHandler`.

Since :class:`ClientTrainer` and :class:`ParameterServerBackendHandler` are required to manipulate PyTorch Model. They are both inherited from :class:`ModelMaintainer`.

.. code-block:: python

    class ModelMaintainer(object):
    """Maintain PyTorch model.

        Provide necessary attributes and operation methods.

    Args:
        model (torch.Module): PyTorch model.
        cuda (bool): use GPUs or not.
    """
    def __init__(self, model, cuda) -> None:
        
        self.cuda = cuda

        if cuda:
            # dynamic gpu acquire. 
            self.gpu = get_best_gpu()
            self._model = model.cuda(self.gpu)
        else:
            self._model = model.cpu()

    @property
    def model(self):
        """Return torch.nn.module"""
        return self._model

    @property
    def model_parameters(self):
        """Return serialized model parameters."""
        return SerializationTool.serialize_model(self._model)

    @property
    def shape_list(self):
        """Return shape of parameters"""
        shape_list = [param.shape for param in self._model.parameters()]
        return shape_list

Client local training
=======================

The basic class of ClientTrainer is shown below, we encourage users define local training process following our code pattern:

.. code-block:: python

    class ClientTrainer(ModelMaintainer):
        """An abstract class representing a client backend trainer.

        In our framework, we define the backend of client trainer show manage its local model.
        It should have a function to update its model called :meth:`local_process`.

        If you use our framework to define the activities of client, please make sure that your self-defined class
        should subclass it. All subclasses should overwrite :meth:`local_process`.

        Args:
            model (torch.nn.Module): PyTorch model.
            cuda (bool): Use GPUs or not.
        """

        def __init__(self, model, cuda):
            super().__init__(model, cuda)
            self.client_num = 1  # default is 1.
            self.type = ORDINARY_TRAINER

        @property
        def uplink_package(self):
            """Return a tensor list for uploading to server.

                This attribute will be called by client manager.
                Customize it for new algorithms.
            """
            return [self.model_parameters]

        def local_process(self, payload):
            """Manager of the upper layer will call this function with accepted payload"""
            raise NotImplementedError()
            
        def train(self):
            """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
            raise NotImplementedError()

        def evaluate(self):
            """Evaluate quality of local model."""
            raise NotImplementedError()


- Overwrite :meth:`ClientTrainer.local_process()` to define local procedure. Typically, you need to implement standard training pipeline of PyTorch.
- Attributes ``model`` and ``model_parameters`` is is associated with ``self._model``. Please make sure the function ``local_process()`` will manipulate ``self._model``.

**A standard implementation of this part is in :class:`SGDClientTrainer`.**

Server global aggregation
==========================

Calculation tasks related with PyTorch should be define in ServerHandler part. In **FedLab**, our basic class of Handler is defined in :class:`ParameterServerBackendHandler`.

.. code-block:: python

   class ParameterServerBackendHandler(ModelMaintainer):
        """An abstract class representing handler of parameter server.

        Please make sure that your self-defined server handler class subclasses this class

        Example:
            Read source code of :class:`SyncParameterServerHandler` and :class:`AsyncParameterServerHandler`.
        """

        def __init__(self, model, cuda=False):
            super().__init__(model, cuda)

        @property
        def downlink_package(self):
            """Property for manager layer. Server manager will call this property when activates clients."""
            return [self.model_parameters]

        @property
        def if_stop(self):
            """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
            return False

        def _update_global_model(self, *args, **kwargs):
            """Override this function for iterating global model (aggregation or optimization)."""
            raise NotImplementedError()

User can define server aggregation strategy by finish following functions:

- You can overwrite ``_update_global_model()`` to customize global procedure.

- ``_update_global_model()`` is required to manipulate global model parameters (self._model).

- Summarised FL aggregation strategies are implemented in ``fedlab.utils.aggregator``.

**A standard implementation of this part is in SyncParameterServerHandler.**

