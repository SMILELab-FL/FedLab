# Benchmarks


## FedAvg


## FedAsgd


## Leaf

### 下载并划分数据

> LEAF benchmark 包含`celeba`, `femnist`, `reddit`, `sent140`, `shakespeare`, `synthetic`六类数据集的联邦设置。
>
> 子文件夹`datasets/leaf_data`中包含了LEAF中各数据集的下载和预处理脚本，来源于[LEAF-Github](https://github.com/TalwalkarLab/leaf)。

- 进入子文件夹`datasets/leaf_data`，选择需要下载的数据集文件夹进入，通过运行`preprocess.sh` 并提供一定的参数选择实现数据集的下载与划分，具体参数见于各数据集文件夹中的`README.md`
- 各数据集文件夹下的`download.sh`提供了一种数据划分参数选择示例，通过运行该文件可获得处理后可用于实验的数据集。

**注意事项：**

1. 对于各数据集而言，单独运行该数据集文件夹下的`preprocess.sh` 而不提供相应的参数，则只会获得原始下载数据`raw_data`和处理原数据但未进行划分的`all_data`两类位于`data`文件夹中。
2. 目前实验数据需要提供训练数据和测试数据的划分，即`data`文件夹下需要存在`train`和`test`文件夹保存相应的数据。
   训练数据和测试数据的划分可通过为`preprocess.sh` 提供相应的参数来实现，如：`-t`参数指定划分用户到训练-测试集合中的方法，`-tf`参数指定训练集的数据比例。
3. 若需要重新获取数据或划分数据，则需要先删除`data`文件夹再运行数据获取或划分脚本。

### 运行实验

- 当前leaf数据集所进行的实验为FedAvg联邦平均算法的实现，实验代码位于`fedavg`文件夹中。目前leaf数据集的fedavg实验完成了`femnist`和`shakespeare`两类。
- `fedavg`文件夹下的`run_leaf_test.sh`脚本包含了leaf数据集客户端数的小规模模拟，通过各数据集名称和提供的总进程数，创建对应的服务器进程和剩余客户端进程，并进行实验。***（具体进程创建脚本可参见##FedAvg中运行脚本说明）***