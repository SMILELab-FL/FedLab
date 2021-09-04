# Benchmarks

FedLab provides PyTorch implementations of common federated learning benchmarks, including Synchronous Federated Algorithm (FedAvg), Asynchronous Federated Algorithm (FedAsgd), Communication Compression Strategy (top-K & DGC), Federated DataSet benchmark (LEAF).

More implementation of algorithms will be provided in the future.  
Read this in another language: [简体中文](./README_zh-cn.md).
## FedAvg

FedAvg is the baseline of synchronous federated learning algorithm, and FedLab implements the algorithm flow of FedAvg, including standalone and Cross Machine scenarios.

### Standalone

The` SerialTrainer` module is responsible for the simulation of single machine and single process, and its source code can be found in `fedlab/core fedlab/core/client/trainer.py`.

Executable scripts is in ` fedlab_benchmarks/algorithm/fedavg/standalone/`.

### Cross Machine

The federated simulation of **multi-machine multi-process** and **single-machine multi-process** scenarios is the core module of FedLab, which is composed of various modules in `core/client` and `core/server`, please refer to overview for details .

The executable script is in `fedlab_benchmarks/algorithm/fedavg/cross_machine/`

## Leaf

**FedLab migrates the TensorFlow version of LEAF dataset to the PyTorch framework, and provides the implementation of dataloader for the corresponding dataset. The unified interface is in `fedlab_benchmarks/dataset/leaf_data_process/dataloader.py`**

### description of Leaf datasets

The LEAF benchmark contains the federation settings of Celeba, femnist, Reddit, sent140, shakespeare and synthetic datasets. With reference to [leaf-readme.md](https://github.com/talwalkarlab/leaf) , the introduction the total number of users and the corresponding task categories of leaf datasets are given below.

1. FEMNIST

- **Overview:** Image Dataset
- **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
- **Task:** Image Classification

2. Sentiment140

- **Overview:** Text Dataset of Tweets
- **Details** 660120 users
- **Task:** Sentiment Analysis

3. Shakespeare

- **Overview:** Text Dataset of Shakespeare Dialogues
- **Details:** 1129 users (reduced to 660 with our choice of sequence length. See [bug](https://github.com/TalwalkarLab/leaf/issues/19).)
- **Task:** Next-Character Prediction

4. Celeba

- **Overview:** Image Dataset based on the [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Details:** 9343 users (we exclude celebrities with less than 5 images)
- **Task:** Image Classification (Smiling vs. Not smiling)

5. Synthetic Dataset

- **Overview:** We propose a process to generate synthetic, challenging federated datasets. The high-level goal is to create devices whose true models are device-dependant. To see a description of the whole generative process, please refer to the paper
- **Details:** The user can customize the number of devices, the number of classes and the number of dimensions, among others
- **Task:** Classification

6. Reddit

- **Overview:** We preprocess the Reddit data released by [pushshift.io](https://files.pushshift.io/reddit/) corresponding to December 2017.
- **Details:** 1,660,820 users with a total of 56,587,343 comments.
- **Task:** Next-word Prediction.

### Download and preprocess data

> For the six types of leaf datasets, refer to [leaf/data](https://github.com/talwalkarlab/leaf/tree/master/data) and provide data download and preprocessing scripts in `fedlab _ benchmarks/datasets/data`.

Common structure of leaf dataset folders:

```
/FedLab/fedlab_benchmarks/dataset/data/{leaf_dataset}

   ├── {other_useful_preprocess_util}
   ├── create_datasets_and_save.sh
   ├── prerpocess.sh
   ├── stats.sh
   └── README.md
```
- `preprocess.sh`: downloads and preprocesses the dataset
- `create_datasets_and_save.sh`: encapsulates the use of `preprocess.sh`, and processes each client' data into the corresponding Dataset, which is stored in the form of a pickle file
- `stats.sh`: performs information statistics on all data (stored in `./data/all_data/all_data.json`) processed by `preprocess.sh`
- `README.md`: gives a detailed description of the process of downloading and preprocessing the dataset, including parameter descriptions and precautions.

> **Developers can directly run the executable script `create_datasets_and_save.sh` to obtain the dataset, process and store the corresponding dataset data in the form of a pickle file.**
> This script provides an example of using the preprocess.sh script, and developers can modify the parameters according to application requirements.

**preprocess.sh Script usage example:**

```shell
cd data/femnist
bash preprocess.sh -s niid --sf 0.05 -k 0 -t sample
cd data/shakespeare
bash preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8
cd data/sent140
bash ./preprocess.sh -s niid --sf 0.05 -k 3 -t sample
cd data/celeba
bash ./preprocess.sh -s niid --sf 0.05 -k 5 -t sample
cd data/synthetic
bash ./preprocess.sh -s niid --sf 1.0 -k 5 -t sample --tf 0.6

# for reddit, see its README.md to download preprocessed dataset manually
```

By setting parameters for `preprocess.sh`, the original data can be sampled and spilted. **The `readme. md` in each dataset folder provides the example and explanation of script parameters, the common parameters are: **

1. `-s` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d. is included in the 'Notes' section
2. `--sf` := fraction of data to sample, written as a decimal; default is 0.1
3. `-k ` := minimum number of samples per user
4. `-t` := 'user' to partition users into train-test groups, or 'sample' to partition each user's samples into train-test groups
5. `--tf` := fraction of data in training set, written as a decimal; default is 0.9

At present, FedLab's Leaf experiment need provided training data and test data, so **we needs to provide related data training set-test set splitting parameter for `preprocess.sh`** to carry out the experiment.

**If you need to obtain or split data again, make sure to delete `data` folder in the dataset directory before re-running `preprocess.sh` to download and preprocess data.**

### Dates usage - dataloader

Leaf datasets are loaded by `dataloader.py` (located under fedlab_benchmarks/dataset/leaf_data_process/). All the returned data are in the form of pytorch [Dataloader](https://pytorch.org/docs/stable/data.html).

### Run experiment


At present, the experiment of LEAF dataset is in **single-machine multi-process** scenario under FedAvg's Cross Machine, and the tests of femnist and Shakespeare datasets have been completed.

The executable script is located in `fedlab_benchmarks/fedavg/cross_machine/run_leaf_test.sh`, which contains a small-scale simulation of the LEAF data set client. According to dataset name and the total number of processes provided, the corresponding server processes and the remaining client processes are created and experimented.