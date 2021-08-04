# FedLab: A Flexible Federated Learning Framework

[![Documentation Status](https://readthedocs.com/projects/fedlab-fedlab/badge/?version=latest&token=24c27118c61cc32da390946ad541028871fb336025d47404d1b6be000727ac4a)](https://fedlab-fedlab.readthedocs-hosted.com/en/latest/?badge=latest)

_其他语言版本：[English](README.md), [简体中文](README.zh-cn.md)._

## 目录

- [简介](#简介)
- [框架设计](#框架设计)
  - [服务器端](#服务器端)
  - [客户端](#客户端)
  - [通信](#通信)
- [部署场景](#部署场景)
  - [单机模式](#单机模式)
  - [跨机模式](#跨机模式)
  - [跨局域网模式](#跨局域网模式)
- [如何使用?](#如何使用)
  - [快速开始](#快速开始)
  - [文档](#文档)
- [引用](#引用)
- [如何贡献代码](#如何贡献代码)
- [联系方式](#联系方式)



## 简介

__FedLab__是一个基于PyTorch的轻量级、组件化联邦学习框架，帮助使用者在单机或多机环境下快速实现联邦学习算法的模拟。 
框架分为server和client两部分，server和client都由`NetworkManager`模块用于通信和消息处理，`NetworkManager`模块基于[torch.distributed](https://pytorch.org/docs/stable/distributed.html)实现的分布式点对点通信模块，负责消息处理和调用后端。
server的后端计算逻辑由`ParameterServerHandler`负责，client的后端计算由`Trainer`负责。Manager模块构成通信协议和压缩框架，`ParameterServe`和`Trainer`构成联邦学习和优化框架。  

__FedLab__提供了一系列构建联邦学习系统的组件和demo，主要分为同步联邦学习和异步联邦学习，并实现了常见的联邦学习算法benchmarks。

更多细节请参考我们的[论文](https://arxiv.org/abs/2107.11621).



## 框架设计

<img src="./docs/imgs/fedlab-overview.svg?raw=True" width=600>

### 服务器端

server端`NetworkManager`与`ParameterServerHandler`的关系如下图，`NetworkManager`处理信息并调用`ParameterServerHandler.on_receive()`方法，`ParameterServerHandler`处理上层调用并更新全局模型(
Global Model)。

<img src="./docs/imgs/fedlab-server.svg?raw=True" width=450>

### 客户端

client端架构和各模块功能类似于server端，但`NetworkManager`和`Trainer`的功能和处理细节不同。client端后端统一为`Trainer`，向上层提供底层模型的训练算法调用，用于定义torch模型训练流程。`NetworkManager`管理前后端逻辑协调和消息处理。

<img src="./docs/imgs/fedlab-client.svg?raw=True" width=450>

### 通信

其中，异步和同步联邦的`NetworkManager`通信逻辑如下图。

1. 同步联邦学习中，一轮学习的启动由server主导，即server执行参与者采样（sample clients），广播全局模型参数。

   <img src="./docs/imgs/fedlab-sychronous.svg?raw=True" width=500>

2. 异步联邦中由client主导，即client向联邦服务器请求当前模型参数，进行本地模型训练。

<img src="./docs/imgs/fedlab-asychronous.svg?raw=True" width=500>

## 部署场景

__FedLab__支持单机、跨机，以及跨局域网联邦学习系统的部署和模拟。

### 单机模式

串行训练器，使用一个进程资源进程多client联邦模拟：

<img src="./docs/imgs/fedlab-SerialTrainer.svg?raw=True" width=450>

### 跨机模式

多进程模拟，在一台机器或多个机器上执行多个联邦脚本：

<img src="./docs/imgs/fedlab-multi_process.svg?raw=True" width=450>

### 跨局域网模式

分层联邦通信，添加`Scheduler`做消息转发，构建跨局域网域联邦，或自定义`Scheduler`功能作为中间服务器，构成负载均衡，满足扩展性，可用于大规模联邦学习模拟。同时`Scheduler`满足跨局域网消息转发的功能，因此__FedLab__支持跨局域网联邦。

<img src="./docs/imgs/fedlab-hierarchical.svg?raw=True" width=600>



## 如何使用?

### 快速开始

1. 配置python环境：

```shell
pip install -r requirements.txt
```

2. 进入./fedlab_benchmarks/algorithm/fedavg/， 运行FedAvg demo：

```shell
bash run.sh 
```

### 文档

[FedLab Docs (prototype)](https://fedlab.readthedocs.io/en/latest/)



## 引用

如果你的工作用到了FedLab，请引用

```bibtex
@article{smile2021fedlab,  
    title={FedLab: A Flexible Federated Learning Framework},  
    author={Dun Zeng, Siqi Liang, Xiangjing Hu and Zenglin Xu},  
    journal={arXiv preprint arXiv:2107.11621},  
    year={2021}
}
```



## 如何贡献代码

欢迎提交pull request贡献代码。

- 代码应遵循[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)协议
- 代码注释规范遵守docstring规范[中文版](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)|[English](https://google.github.io/styleguide/pyguide.html)



## 联系方式

请通过GitHub issues或邮件联系__FedLab__开发团队：

- Dun Zeng: zengdun@foxmail.com
- Siqi Liang: zszxlsq@gmail.com

