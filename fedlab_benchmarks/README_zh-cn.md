# Benchmarks

FedLab 提供了常见的联邦学习基准方法的PyTorch实现包括：同步联邦算法（FedAvg）、异步联邦算法（FedAsgd）、通信压缩策略（Top-k & DGC）、联邦数据集（LEAF）。更多的算法的FedLab实现会在未来提供。

## FedAvg

FedAvg是同步联邦学习算法的baseline，FedLab实现了FedAvg的算法流程，包括standalone和cross machine的场景。

### Standalone

单机单进程模拟由`SerialTrainer`模块负责，其源代码可见`fedlab/core/client/trainer.py`，
可执行的脚本在`fedlab_benchmarks/algorithm/fedavg/standalone/`。

### Cross Machine

**多机多进程**和**单机多进程**场景的联邦模拟是FedLab的核心模块，由`core/client`和`core/server`中的各项模块组成，具体细节请见overview。

可执行的脚本在`fedlab_benchmarks/algorithm/fedavg/cross_machine/`

## Leaf

**FedLab将TensorFlow版本的LEAF数据集迁移到了PyTorch框架下，并提供了相应数据集的dataload的实现脚本，统一的接口在`fedlab_benchmarks/dataset/leaf_data_process/dataloader.py`**

### 下载并划分数据

> `fedlab_benchmarks/datasets`提供了常用数据集的下载脚本，通过运行脚本可获得相应数据。
>
> LEAF benchmark 包含了celeba, femnist, reddit, sent140, shakespeare, synthetic 六类数据集的联邦设置，在`fedlab_benchmarks/datasets`提供了相关的数据下载和预处理脚本，参考[LEAF-Github](https://github.com/TalwalkarLab/leaf) 
>
> **对于LEAF实验，FedLab为每个数据集提供了一种数据下载、采样和划分参数示例脚本，位于`fedlab_benchmarks/datasets/{dataset_name}/download.sh`，运行后可直接用于实验。**



**以下对脚本使用进行说明：**

进入`fedlab_benchmarks/datasets/{dataset_name}`，运行`preprocess.sh` 得到相关数据并存储于`./data`文件夹，包含原始下载数据`raw_data`和处理后的全部数据`all_data`两类。

- [LEAF - README.md](https://github.com/TalwalkarLab/leaf) 给出了六类数据集的简介、总用户数和对应任务类别，如FEMNIST数据集是用于图像识别的图像数据集，总类别包含大小写字母和数字在内共62个类，每张图像大小为28x28，共有3500个用户。
- 下载完数据后，开发者可对各数据集运行`./stas.sh`得到`./data/all_data/all_data.json`的统计信息。
- 若提供相关参数可对原始数据进行采样、划分，实现数据的联邦应用设置，**常用参数有**：
  1. ```-s```表示采样方式，取值有'iid'和'niid'两种选择，表示是否使用i.i.d方式进行采样；
  2. ```--sf```表示采样数据比例，取值为小数，默认为0.1；
  3. ```-k``` 表示采样时所要求的用户最少样本数目，筛选掉拥有过少样本的用户，若取值为0表示不进行样本数目的筛选。
  4. ```-t```表示划分训练集测试集的方式，取值为'user'则划分用户到训练-测试集合，取值为'sample'则划分每个用户的数据到训练-测试集合中；
  5. ```--tf``` 表示训练集的数据占比，取值为小数，默认为0.9，表示训练集:测试集=9:1。

目前FedLab对LEAF六类数据集的实验需要提供训练数据和测试数据，因此**需要对`preprocess.sh`提供相关的数据划分参数**。

- **`./download.sh`为开发者提供了一种数据采样和划分参数示例用于实验**，使用者可运行该脚本获取处理后的训练集和测试集，存储于`./data/train`和`./data/test`中。
  如：```bash preprocess.sh -s niid --sf 0.05 -k 0 -t sample```

- 若需要重新获取数据或划分数据，需要先删除`data`文件夹再运行相关脚本进行数据下载和处理。

### 运行实验

当前LEAF数据集所进行的实验为FedAvg的cross machine下的和**单机多进程**场景，目前已完成femnist和shakespeare两类数据集的测试。

可执行脚本位于 `fedlab_benchmarks/fedavg/cross_machine/run_leaf_test.sh`，包含了LEAF数据集客户端的小规模模拟，通过各数据集名称和提供的总进程数，创建对应的服务器进程和剩余客户端进程，并进行实验。