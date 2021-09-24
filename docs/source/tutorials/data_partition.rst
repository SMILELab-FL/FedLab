.. _data-partition:

****************
Data Partitioner
****************

This chapter introduces the dataset partitioner ``DataPartitioner`` and how the client process uses the corresponding dataset. **FedLab** provides various methods to deal with different partition strategy corresponding with different dataset situations.

For classification datasets, FedLab provides noniid and random partition method.

CIFAR10Partitioner
==================





In codes above, data\_indices is a dictionary (in order to ensure that the partition result is consistent in the case of cross-machine, the user can save the division dictionary in a file) like this：

.. code:: python

    dict= { '0': indices of dataset,
            '1': indices of dataset,
            ...
            'k': indices of dataset }

By using torch's sampler, only the right part of the sample is taken from the overall dataset.

.. code:: python

    from fedlab.utils.dataset.sampler import SubsetSampler

    train_loader = torch.utils.data.DataLoader(
                    trainset,
                    sampler=SubsetSampler(indices=data_slices[client_id],
                                          shuffle=True),
                    batch_size=batch_size)

There is also a similar implementation of directly reordering and partition the dataset, see fedlab.utils.dataset.sampler.RawPartitionSampler for details.

In addition to dividing the dataset by the sampler of torch, dataset can also be divided directly by splitting the dataset file. The implementation can refer to FedLab version of LEAF.
