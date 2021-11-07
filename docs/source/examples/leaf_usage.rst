.. _leaf:

***********************
PyTorch version of LEAF
***********************

**FedLab migrates the TensorFlow version of LEAF dataset to the PyTorch
framework, and provides the implementation of dataloader for the
corresponding dataset. The unified interface is in
``fedlab_benchmarks/leaf/dataloader.py``**

This markdown file introduces the process of using LEAF dataset in
FedLab.

Description of Leaf datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LEAF benchmark contains the federation settings of Celeba, femnist, Reddit, sent140, shakespeare and synthetic datasets. With reference to `leaf-readme.md <https://github.com/talwalkarlab/leaf>`__ , the introduction the total number of users and the corresponding task categories of leaf datasets are given below.

1. FEMNIST

-  **Overview:** Image Dataset.
-  **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users.
-  **Task:** Image Classification.

2. Sentiment140

-  **Overview:** Text Dataset of Tweets.
-  **Details** 660120 users.
-  **Task:** Sentiment Analysis.

3. Shakespeare

-  **Overview:** Text Dataset of Shakespeare Dialogues.
-  **Details:** 1129 users (reduced to 660 with our choice of sequence length. See `bug <https://github.com/TalwalkarLab/leaf/issues/19>`__.)
-  **Task:** Next-Character Prediction.

4. Celeba

-  **Overview:** Image Dataset based on the `Large-scale CelebFaces Attributes Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__.
-  **Details:** 9343 users (we exclude celebrities with less than 5 images).
-  **Task:** Image Classification (Smiling vs. Not smiling).

5. Synthetic Dataset

-  **Overview:** We propose a process to generate synthetic, challenging federated datasets. The high-level goal is to create devices whose true models are device-dependant. To see a description of the whole generative process, please refer to the paper.
-  **Details:** The user can customize the number of devices, the number of classes and the number of dimensions, among others.
-  **Task:** Classification.

6. Reddit

-  **Overview:** We preprocess the Reddit data released by `pushshift.io <https://files.pushshift.io/reddit/>`__ corresponding to December 2017.
-  **Details:** 1,660,820 users with a total of 56,587,343 comments.
-  **Task:** Next-word Prediction.

Download and preprocess data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    For the six types of leaf datasets, refer to `leaf/data <https://github.com/talwalkarlab/leaf/tree/master/data>`__ and provide data download and preprocessing scripts in ``fedlab _ benchmarks/datasets/data``. In order to facilitate developers to use leaf, fedlab integrates the download and processing scripts of leaf six types of data sets into ``fedlab_benchmarks/datasets/data``, which stores the download scripts of various data sets.

Common structure of leaf dataset folders:

::

    /FedLab/fedlab_benchmarks/datasets/{leaf_dataset_name}

       ├── {other_useful_preprocess_util}
       ├── prerpocess.sh
       ├── stats.sh
       └── README.md

-  ``preprocess.sh``: downloads and preprocesses the dataset
-  ``stats.sh``: performs information statistics on all data (stored in ``./data/all_data/all_data.json``) processed by ``preprocess.sh``
-  ``README.md``: gives a detailed description of the process of downloading and preprocessing the dataset, including parameter descriptions and precautions.

    **Developers can directly run the executable script ``create_datasets_and_save.sh`` to obtain the dataset, process and store the corresponding dataset data in the form of a pickle file.** This script provides an example of using the preprocess.sh script, and developers can modify the parameters according to application requirements.

**preprocess.sh Script usage example:**

.. code:: shell

    cd fedlab_benchmarks/datasets/data/femnist
    bash preprocess.sh -s niid --sf 0.05 -k 0 -t sample

    cd fedlab_benchmarks/datasets/data/shakespeare
    bash preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8

    cd fedlab_benchmarks/datasets/data/sent140
    bash ./preprocess.sh -s niid --sf 0.05 -k 3 -t sample

    cd fedlab_benchmarks/datasets/data/celeba
    bash ./preprocess.sh -s niid --sf 0.05 -k 5 -t sample

    cd fedlab_benchmarks/datasets/data/synthetic
    bash ./preprocess.sh -s niid --sf 1.0 -k 5 -t sample --tf 0.6

    # for reddit, see its README.md to download preprocessed dataset manually

By setting parameters for ``preprocess.sh``, the original data can be sampled and spilted. The ``readme.md`` in each dataset folder provides the example and explanation of script parameters, the common parameters are: 

1. ``-s`` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d. is included in the 'Notes' section.

2. ``--sf`` := fraction of data to sample, written as a decimal; default is 0.1.

3. ``-k`` := minimum number of samples per user

4. ``-t`` := 'user' to partition users into train-test groups, or 'sample' to partition each user's samples into train-test groups

5. ``--tf`` := fraction of data in training set, written as a decimal; default is 0.9, representing train set: test set = 9:1.

At present, FedLab's Leaf experiment need provided training data and test data, so we needs to provide related data training set-test set splitting parameter for ``preprocess.sh`` to carry out the experiment, default is 0.9.

If you need to obtain or split data again, make sure to delete ``data`` folder in the dataset directory before re-running ``preprocess.sh`` to download and preprocess data.

Pickle file stores Dataset.
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to speed up developers' reading data, fedlab provides a method of processing raw data into Dataset and storing it as a pickle file. The Dataset of the corresponding data of each client can be obtained by reading the pickle file after data processing.

set the parameters and run ``create_pickle_dataset.py``. The usage example is as follows:

.. code:: shell

    cd fedlab_benchmarks/leaf/process_data
    python create_pickle_dataset.py --data_root "../../datasets" --save_root "./pickle_dataset" --dataset_name "shakespeare"

Parameter Description: 

1. ``data_root`` : the root path for storing leaf data sets, which contains all leaf data sets; If you use the ``Fedlab_benchmarks/datasets/`` provided by fedlab to download leaf data, 'data\_root' can be set to this path, a relative address of which is shown in this example. 

2. ``save_root``: directory to store the pickle file address of the processed Dataset; Each dataset Dataset will be saved in ``{save_root}/{dataset_name}/{train,test}``; the example is to create a ``pickle_dataset`` folder under the current path to store all pickle dataset files. 

3. ``dataset_name``: Specify the name of the leaf data set to be processed. There are six options {femnist, shakespeare, celeba, sent140, synthetic, reddit}.

Dataloader loading data set
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leaf datasets are loaded by ``dataloader.py`` (located under ``fedlab_benchmarks/leaf/dataloader.py``). All returned data types are pytorch `Dataloader <https://pytorch.org/docs/stable/data.html>`__.

By calling this interface and specifying the name of the data set, the corresponding Dataloader can be obtained.

**Example of use:**

.. code:: python

    from leaf.dataloader import get_LEAF_dataloader
    def get_femnist_shakespeare_dataset(args):
        if args.dataset == 'femnist' or args.dataset == 'shakespeare':
            trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset,
                                                          client_id=args.rank)
        else:
            raise ValueError("Invalid dataset:", args.dataset)

        return trainloader, testloader

Run experiment
~~~~~~~~~~~~~~

The current experiment of LEAF data set is the **single-machine multi-process** scenario under FedAvg's Cross machine implement, and the tests of femnist and Shakespeare data sets have been completed.

Run \`fedlab\_benchmarks/fedavg/cross\_machine/LEAF\_test.sh\` to quickly execute the simulation experiment of fedavg under leaf data set.
