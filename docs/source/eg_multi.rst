
Cross Machine/Process
=====================
In this page, we introduce how to build a FL simulation system with FedLab in cross machine or cross process scenario. We implement FedAvg algorithm wit CNN and partitioned MNIST dataset across clients.
Source code of this page can be seen in fedlab_benchamrks/algorithm/fedavg/cross_machine.



Client related part
^^^^^^^^^^^^^^^^^^^^^
Network Configuration
----------------------
Netowrk configuration basiclly follows the Examples page. Please make sure the right IP/PORT setting. The ethernet can be None, torch.distributed will find the right ethernet to use. But when it doesn't work, user need to assign right ethernet name (shown by ifconfig).  

Due to the rank of torch.distributed is unique for every process. Therefore, we use rank represent client id for this scenario. Typically, world size = the number of client + 1(server).


Dataset Partition
------------------
User need to define a dataloader for Trainer. We encourage you use torch.utils.data.Sampler to define your partition strategy.
FedLab implemented some useful functions to build dataset partition. Related sourece codes or docs can be seen in fedlab/utils/dataset

For example, regular classification dataset can be partitioned as follows:

.. code-block:: python

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    # trainset = torchvision.datasets.MNIST(root=root, train=True, download=True)

    data_indices = noniid_slicing(trainset, num_clients=100, num_shards=200)
    save_dict(data_indices, "cifar10_noniid.pkl")

    data_indices = random_slicing(trainset, num_clients=100)
    save_dict(data_indices, "cifar10_iid.pkl")

data_indices is a dict map from client id to data indices(list) of raw dataset. FedLab provides random partition and noniid partition methods, in which the noniid partition method is totally reimplementation in paper fedavg.

Trainer
--------
User can define any train procedure with pytorch in this part. But some interface must be given by trainer in FedLab framework which is shown by the base class ClientTrainer.

User can overwrite the ``ClientTrainer.train()`` to manipulate ``self._model``. The attributes of ``model`` and ``model_parameters`` must be given to upper layer as well.

Standard trainer implementation as follows:

.. code-block:: python

    class ClientSGDTrainer(ClientTrainer):
    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=True,
                 logger=None):
        super(ClientSGDTrainer, self).__init__(model, cuda)

        self._data_loader = data_loader

        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def train(self, model_parameters, epochs=None) -> None:
        self._LOGGER.info("Local train procedure is started")
        SerializationTool.deserialize_model(self._model, model_parameters)  # load parameters
        if epochs is None:
            epochs = self.epochs
        self._LOGGER.info("Local train procedure is running")
        for _ in range(epochs):
            self._model.train()
            for inputs, labels in self._data_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(self.gpu)

                outputs = self._model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")


Client Manager
---------------
The basic class of NetworkManager is placed in fedlab/core/network_manager.py. Manager is core part in FedLab, organizing control flow and communication flow.

Standard implementation is shown below:

.. code-block:: python

    class ClientPassiveManager(NetworkManager):
        def __init__(self, handler, network, logger=None):
            super(ClientPassiveManager, self).__init__(network, handler)

            if logger is None:
                logging.getLogger().setLevel(logging.INFO)
                self._LOGGER = logging
            else:
                self._LOGGER = logger

        def run(self):
            self._LOGGER.info("connecting with server")
            self.setup()
            while True:
                self._LOGGER.info("Waiting for server...")
                # waits for data from server (default server rank is 0)
                sender_rank, message_code, payload = PackageProcessor.recv_package(
                    src=0)
                # exit
                if message_code == MessageCode.Exit:
                    self._LOGGER.info(
                        "Receive {}, Process exiting".format(message_code))
                    self._network.close_network_connection()
                    break
                else:
                    # perform activation strategy
                    self.on_receive(sender_rank, message_code, payload)

                # synchronize with server
                self.synchronize()

        def on_receive(self, sender_rank, message_code, payload):

            self._LOGGER.info("Package received from {}, message code {}".format(
                sender_rank, message_code))
            model_parameters = payload[0]
            self._handler.train(model_parameters=model_parameters)

        def setup(self):
            self._network.init_network_connection()

.. note::

    1. ``setup()`` defines the network initialization stage. Can be used in complex system information synchronize.
    2. ``run()`` is the main process of client. User need to define the communication strategy with user. 
    3. ``on_receive(sender_rank, message_code, payload)`` indicate the control flow and information parsing.




Server related part
^^^^^^^^^^^^^^^^^^^^^


Network Configuration
----------------------
Network Configuration in Server is the same as client. But please be aware of that we assume that the rank of server is 0 as default.

Control Flow
-------------
Unlike client part, the control flow of Server can be more complicated. Typically, Sever needs to define all related communication package details and control flow strategy including functions like ``activate_clients``, ``shutdown_clients``.

Backend Handler Strategy
-------------------------
Calculation tasks related with PyTorch should be define in ServerHandler part. In FedLab, our basic class of Handler is defined in ParameterServerBackendHandler. User need to overwrite ``update_model`` to define aggregation strategy and manipulate global model parameters at the same time.

The standard implementation of this part can be seen in SyncParameterServerHandler.