# FedLab: A Flexible Federated Learning Framework

[![Documentation Status](https://readthedocs.com/projects/fedlab-fedlab/badge/?version=latest&token=24c27118c61cc32da390946ad541028871fb336025d47404d1b6be000727ac4a)](https://fedlab-fedlab.readthedocs-hosted.com/en/latest/?badge=latest)

_Read this in other languages: [English](README.md), [简体中文](README.zh-cn.md)._

## Table of Contents



## Introduction

Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine learning, which aims to protect individual data privacy in distributed machine learning process, especially in ﬁnance, smart healthcare and edge computing. Different from traditional data-centered distributed machine learning, participants in FL setting utilize localized data to train local model, then leverages speciﬁc strategies with other participants to acquire the ﬁnal model collaboratively, avoiding direct data sharing behavior.

To relieve the burden of researchers in implementing FL algorithms and emancipate FL scientists from repetitive implementation of basic FL setting, we introduce highly customizable framework __FedLab__ in this work. __FedLab__ provides the necessary modules for FL simulation, including communication, compression, model optimization, data partition and other functional modules. __FedLab__ users can build FL simulation environment with custom modules like playing with LEGO bricks. For better understanding and easy usage, FL algorithm benchmark implemented in __FedLab__ are also presented.

For more details, please read our [full paper](https://arxiv.org/abs/2107.11621).



## Framework Overview

![framework-overview](./docs/imgs/fedlab-overview.pdf?raw=True)

__FedLab__ provides two basic roles in FL setting: `Server` and `Client`. Each `Server`/`Client` consists of two components called `NetworkManager` and `ParameterHandler`/`Trainer`. 

- `NetworkManager` module manages message process task, which provides interfaces to customize communication agreements and compression.
- `ParameterHandler` is responsible for backend computation in `Server`; and `Trainer` is for backend computation in `Client`






### Server

The connection between `NetworkManager`

server端NetworkManager与ParameterServer的关系如下图，NetworkManager处理信息并调用ParameterServer.on_receive方法，ParameterServer处理上层调用并更新全局模型(
Global Model)。

![image](./docs/imgs/fedlab-server.png?raw=True)

### Client

client端架构和各模块功能类似于server端，但NetworkManager和Trainer的功能和处理细节不同。client端后端统一为Trainer，向上层提供底层模型的训练算法调用，用于定义torch模型训练流程。NetworkManager管理前后端逻辑协调和消息处理。

![image](./docs/imgs/fedlab-client.png?raw=True)

### Communication

其中，异步和同步联邦的Network Manager通信逻辑如下图。

1. 同步联邦学习中，一轮学习的启动由server主导，即server执行参与者采样（sample clients），广播全局模型参数。

![同步通信](./docs/imgs/fedlab-sychronous.pdf?raw=True)

2. 异步联邦中由client主导，即client向联邦服务器请求当前模型参数，进行本地模型训练。

![异步通信](./docs/imgs/fedlab-asychronous.pdf?raw=True)

## Experiment Scene

FedLab支持多机和单机联邦学习系统的部署和模拟。

### Standalone

串行训练器，使用一个进程资源进程多client联邦模拟：
![image](./docs/imgs/fedlab-SerialTrainer.pdf?raw=True)

### Cross-Machine

多进程模拟，在一台机器或多个机器上执行多个联邦脚本：
![image](./docs/imgs/fedlab-multi_process.pdf?raw=True)

### Hierarchical

分层联邦通信，添加scheduler做消息转发，构建跨局域网域联邦，或自定义scheduler功能作为middle-server，构成负载均衡，满足扩展性，可用于大规模联邦学习模拟。同时scheduler满足跨局域网消息转发的功能，因此FedLab支持跨域联邦。
![image](./docs/imgs/fedlab-hierarchical.pdf?raw=True)

## How to start FedLab？

### Quick Start

1. Install package dependence:

```shell
pip install -r requirements.txt
```

2. enter `./fedlab_benchmarks/algorithm/fedavg/`, run FedAvg demo:

```shell
# cd into fedlab_benchmarks/algorithm/fedavg/ dir
bash run.sh 
```

### Documentation

[FedLab Docs (prototype)](https://fedlab.readthedocs.io/en/latest/)

## Citation

Please cite __FedLab__ in your publications if it helps your research:

```bibtex
@article{smile2021fedlab,  
    title={FedLab: A Flexible Federated Learning Framework},  
    author={Dun Zeng, Siqi Liang, Xiangjing Hu and Zenglin Xu},  
    journal={arXiv preprint arXiv:2107.11621},  
    year={2021}
}
```



## Contribution Guideline

You're welcome to contribute to this project through _Pull Request_.

- By contributing, you agree that your contributions will be licensed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html) 
- Docstring  and code should follow Google Python Style Guide: [中文版](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)|[English](https://google.github.io/styleguide/pyguide.html)
- The code should provide test cases using `unittest.TestCase`



## Contact

Contact the __FedLab__ development team through Github issues or email: 

- Zeng Dun: zengdun@foxmail.com
- Siqi Liang: zszxlsq@gmail.com

