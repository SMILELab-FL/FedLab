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

### Leaf数据集说明

LEAF benchmark 包含了celeba, femnist, reddit, sent140, shakespeare, synthetic 六类数据集的联邦设置。参考[leaf - README.md](https://github.com/TalwalkarLab/leaf) ，给出六类数据集的简介、总用户数和对应任务类别。

1. FEMNIST
- **概述：** 图像数据集
- **详情：**
  共有62个不同类别（10个数字，26个小写字母，26个大写字母）；
  每张图像是28 * 28像素（可选择全部处理为128 * 128像素）；
  共有3500位用户。
- **任务：** 图像分类

2. Sentiment140

- **概述：** 推特推文文本数据集
- **详情：** 共660120位用户
- **任务：** 情感分析

3. Shakespeare

- **概述：** 莎士比亚对话文本数据集
- **详情：** 共1129位用户（后续根据序列长度减少到660位，详情查看[bug](https://github.com/TalwalkarLab/leaf/issues/19) ）
- **任务：** 下一字符预测

4. Celeba

- **概述：** 基于 [大规模名人面孔属性数据集](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 的图像数据集
- **详情：** 共9343位用户（排除了样本数小于等于5的名人）
- **任务：** 图像识别（微笑检测）

5. Synthetic Dataset

- **概述：** 提出了一个生成具有挑战性的合成联合数据集的过程，高级目标是创建真实模型依赖于各设备的设备。可参阅论文["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097) 查看整个生成过程的描述。
- **详情：** 用户可以自定义设备数量、类别数量和维度数量等
- **任务：** 分类

6. Reddit

- **概述：** 对[pushshift.io](https://files.pushshift.io/reddit/) 发布的2017年12月的Reddit数据进行了预处理
- **详情：** 共1,660,820位用户，总评论56,587,343条。
- **任务：** 下一单词预测

### 下载并处理数据

> 关于leaf六类数据集，参考[leaf/data](https://github.com/TalwalkarLab/leaf/tree/master/data) 在`fedlab_benchmarks/datasets/data`中提供了数据下载和预处理脚本。

leaf数据集文件夹常用结构：

```
/FedLab/fedlab_benchmarks/dataset/data/{leaf_dataset}

   ├── {other_useful_preprocess_util}
   ├── prerpocess.sh
   ├── stats.sh
   └── README.md
```

其中可执行脚本`preprocess.sh`对数据集进行下载和处理，`stats.sh`对`preprocess.sh`处理后所有数据（存储于`./data/all_data/all_data.json`）进行信息统计，`README.md`对该数据集的下载和处理过程进行了详细说明，包含了参数说明和注意事项。

**leaf脚本使用样例：**

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

通过对`preprocess.sh`设定参数，实现对原始数据的采样、划分等处理，**各数据集文件夹下的README.md均提供了脚本参数示例和解释，常见参数有：**

1. `-s`表示采样方式，取值有'iid'和'niid'两种选择，表示是否使用i.i.d方式进行采样；
2. `--sf`表示采样数据比例，取值为小数，默认为0.1；
3. `-k` 表示采样时所要求的用户最少样本数目，筛选掉拥有过少样本的用户，若取值为0表示不进行样本数目的筛选。
4. `-t`表示划分训练集测试集的方式，取值为'user'则划分用户到训练-测试集合，取值为'sample'则划分每个用户的数据到训练-测试集合中；
5. `--tf` 表示训练集的数据占比，取值为小数，默认为0.9，表示训练集:测试集=9:1。

目前FedLab的Leaf实验需要提供训练数据和测试数据，因此**需要对`preprocess.sh`提供相关的数据训练集-测试集划分参数**来进行实验。

**若需要重新获取数据或划分数据，需要先删除各数据集下的`data`文件夹再运行相关脚本进行数据下载和处理。**

### 数据集使用 - dataloader

leaf数据集由`dataloader.py`加载(位于`fedlab_benchmarks/dataset/leaf_data_process`)，所有返回数据类型均为pytorch [Dataloader](https://pytorch.org/docs/stable/data.html) 。

### 运行实验

当前LEAF数据集所进行的实验为FedAvg的cross machine下的**单机多进程**场景，目前已完成femnist和shakespeare两类数据集的测试。

可执行脚本位于 `fedlab_benchmarks/fedavg/cross_machine/run_leaf_test.sh`，包含了LEAF数据集客户端的小规模模拟，通过各数据集名称和提供的总进程数，创建对应的服务器进程和剩余客户端进程，并进行实验。