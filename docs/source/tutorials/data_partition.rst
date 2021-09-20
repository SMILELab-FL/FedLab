**********************
Data Partition Scheme
**********************

This chapter introduces the dataset partition strategy and how the client process uses the corresponding dataset. FedLab provides various methods to deal with different partition strategy corresponding with different dataset situations.

torchvision module provides commonly used datasets. We found that the index of a sample in the CIFAR10 and MNIST datasets remains unchange when the dataset is initialized, so the dataset partition can be created according to the initial indices.

For classification datasets, FedLab provides noniid and random partition method. Take the MNIST dataset as an example:

.. code:: python

    import torchvision
    from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

    root = "datasets/mnist/"
    trainset = torchvision.datasets.MNIST(root=root,
                                          train=True,
                                          download=True)
    # Sort the data set by label, cut into num_shards blocks, and divide them into num_clients evenly.
    data_indices = noniid_slicing(trainset, num_clients=100, num_shards=200)
    save_dict(data_indices, "mnist_noniid.pkl")

    # Randomly evenly divided into num_clients blocks.
    data_indices = random_slicing(trainset, num_clients=100)
    save_dict(data_indices, "mnist_iid.pkl")

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
