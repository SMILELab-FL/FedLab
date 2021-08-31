***************************
Federated Optimization
***************************

Standard FL Optimization contains two parts: 1. local train in client; 2. global aggregation in
server.  Local train and aggregation procedure are customizable in FedLab. You need to define
``ClientTrainer`` and ``ParameterServerBackendHandler``.


Client local training
=======================

The basic class of ClientTrainer is shown beblow, we encourage users define local training process flolowing our code pattern:

.. code-block:: python

  class ClientTrainer(ABC):
      """An abstract class representing a client backend handler.

      In our framework, we define the backend of client handler show manage its local model.
      It should have a function to update its model called :meth:`train`.

      If you use our framework to define the activities of client, please make sure that your self-defined class
      should subclass it. All subclasses should overwrite :meth:`train`.

      Args:
          model (torch.nn.Module): Model used in this federation
          cuda (bool): Use GPUs or not
      """
      def __init__(self, model, cuda):
          self.cuda = cuda

          if self.cuda:
              # dynamic gpu acquire.
              self.gpu = get_best_gpu()
              self._model = model.cuda(self.gpu)
          else:
              self._model = model.cpu()

      @abstractmethod
      def train(self):
          """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
          raise NotImplementedError()

      @property
      def model(self):
          """attribute"""
          return self._model

      @property
      def model_parameters(self):
          """attribute"""
          return SerializationTool.serialize_model(self._model)

.. note::
   - Overwrite ``ClientTrainer.train()`` to define local train procedure. Typically, you need to
     implement standard training pipeline of PyTorch.
   - Attributes ``model`` and ``model_parameters`` is 

**A standard implementation of this part is in ClientSGDTrainer.**

Server global aggregation
==========================

Calculation tasks related with PyTorch should be define in ServerHandler part. In FedLab, our basic class of Handler is defined in ParameterServerBackendHandler.

.. code-block:: python

  class ParameterServerBackendHandler(ABC):
      """An abstract class representing handler of parameter server.

      Please make sure that your self-defined server handler class subclasses this class

      Example:
          Read source code of :class:`SyncParameterServerHandler` and :class:`AsyncParameterServerHandler`.
      """
      def __init__(self, model, cuda=False) -> None:
          self.cuda = cuda
          if cuda:
              self._model = model.cuda()
          else:
              self._model = model.cpu()

      @abstractmethod
      def _update_model(self, model_parameters_list) -> torch.Tensor:
          """Override this function to update global model

          Args:
              model_parameters_list (list[torch.Tensor]): A list of serialized model parameters collected from different clients.
          """
          raise NotImplementedError()

      @abstractmethod
      def stop_condition(self) -> bool:
          """Override this function to tell up layer when to stop process.

          :class:`NetworkManager` keeps monitoring the return of this method, and it will stop all related processes and threads when ``True`` returned.
          """
          raise NotImplementedError()

      @property
      def model(self):
          """Return torch.nn.module"""
          return self._model

      @property
      def model_parameters(self):
          """Return serialized model parameters."""
          return SerializationTool.serialize_model(self._model)


.. note:: 
  User can define server aggregation strategy by finish following functions:
    - You can overwrite ``_update_model(model_parameters_list)`` to
      customize aggregation procedure. Typically, you can define aggregation functions as FedLab.
    - ``_update_model(model_parameters_list)`` is required to manipulate global model parameters (self._model).
    - implemented in ``fedlab.utils.aggregator`` which used in FedLab standard implementations.
    - ``stop_condition()`` return True or False according to your strategy. ServerManager will exit when stop_condition returns True.
   

**A standard implementation of this part is in SyncParameterServerHandler.**
