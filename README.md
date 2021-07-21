# FedLab: A Flexible Federated Learning Framework

[![Documentation Status](https://readthedocs.com/projects/fedlab-fedlab/badge/?version=latest&token=24c27118c61cc32da390946ad541028871fb336025d47404d1b6be000727ac4a)](https://fedlab-fedlab.readthedocs-hosted.com/en/latest/?badge=latest) 
[![codecov](https://codecov.io/gh/SMILELab-FL/FedLab/branch/main/graph/badge.svg?token=4HHB5JCSC6)](https://codecov.io/gh/SMILELab-FL/FedLab)

## Introduction

FedLab是一个基于pytorch的轻量级、组件化联邦学习框架，帮助使用者在单机或多机环境下快速实现联邦学习算法的模拟。  
框架分为server和client两部分，server和client都由Topology模块用于通信和消息处理，Topology模块基于[torch.distributed](https://pytorch.org/docs/stable/distributed.html)实现的分布式点对点通信模块，负责消息处理和调用后端。 server的后端计算逻辑由ParameterServerHandler负责，client的后端计算由trainer负责。Topology模块构成通信协议和压缩框架，ParameterServer和Trainer构成联邦学习和优化框架。  
FedLab提供了一系列构建联邦学习系统的组件和Demo，主要分为同步联邦学习和异步联邦学习，并实现了常见的联邦学习算法Benchmarks。

![image](/docs/imgs/fedlab-overview.png?raw=True)

### Server
server端Topology与ParameterServer的关系如下图，Topology处理信息并调用ParameterServer.on_receive方法，ParameterServer处理上层调用并更新全局模型(Global Model)。  

![image](./docs/imgs/fedlab-server.png?raw=True)


### Client

client端架构和各模块功能类似于server端，但Topology和Trainer的功能和处理细节不同。client端后端统一为Trainer，向上层提供底层模型的训练算法调用，用于定义torch模型训练流程。Topology管理前后端逻辑协调和消息处理。

![image](./docs/imgs/fedlab-client.png?raw=True)  

### Communication
其中，异步和同步联邦的Topology通信逻辑如下图。  
1. 同步联邦学习中，一轮学习的启动由server主导，即server执行参与者采样（sample clients），广播全局模型参数。
2. 异步联邦中由client主导，即client向联邦服务器请求当前模型参数，进行本地模型训练。  

![异步通信](./docs/imgs/fedlab-asychronous.png)
![同步通信](./docs/imgs/fedlab-sychronous.png)
## Experiment Scene
FedLab支持多机和单机联邦学习系统的部署和模拟。

### Standalone

串行训练器，使用一个进程资源进程多client联邦模拟：
![image](./docs/imgs/fedlab-SerialTrainer.png?raw=True)

### Cross Machine
多进程模拟，在一台机器或多个机器上执行多个联邦脚本：
![image](./docs/imgs/fedlab-multi_process.png?raw=True)


### Hierarchical
分层联邦通信，添加scheduler做消息转发或sub-server，满足扩展性，可用于大规模联邦学习模拟。同时scheduler满足跨局域网消息转发的功能，因此FedLab支持跨域联邦。
![image](./docs/imgs/fedlab-hierarchical.png?raw=True)  

## How to start FedLab？


### Quick Start
1. 配置python环境
```shell
pip install -r requirements.txt
```
2. 运行FedAvg demo
```shell
bash run.sh 
```

### Benchmarks

TODO：
FedAvg  
FedAsgd  
Leaf  
Compressor  

### Documentation
文档：https://fedlab-fedlab.readthedocs-hosted.com/en/latest/

## Citation

## Contribution Guidance
欢迎提交pull request贡献代码