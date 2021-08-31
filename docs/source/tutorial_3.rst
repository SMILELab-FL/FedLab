.. _tutorial3:

**********************
Federated Optimization
**********************

Standard FL Optimization contains two parts: 1. local train in client; 2. global aggregation in
server.  Local train and aggregation procedure are customizable in FedLab. You need to define
``ClientTrainer`` and ``ParameterServerBackendHandler``.

.. note::
   - Overwrite ``ClientTrainer.train()`` to define local train procedure. Typically, you need to
     implement standard training pipeline of PyTorch.
   - ``ParameterServerBackendHandler`` defines hyperparameter of FL system such as
     ``stop_condition()``, ``sample_clients()`` and so on.
   - You can overwrite ``ParameterServerBackendHandler._update_model(serialized_params_list)`` to
     customize aggregation procedure. Typically, you can define aggregation functions as FedLab
     implemented in ``fedlab.utils.aggregator`` which used in FedLab standard implementations.

.. code-block:: python

    # ClientTrainer
    trainer = ClientSGDTrainer(model, trainloader, epochs, optimizer, criterion, cuda, logger)

    # ParameterServerBackendHandler
    handler = SyncParameterServerHandler(model, client_num_in_total, global_round, logger, sample_ratio)


Client local training
=======================



Server global aggregation
==========================
