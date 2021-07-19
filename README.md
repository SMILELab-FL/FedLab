# FedLab: A Flexible Federated Learning Framework

[![Documentation Status](https://readthedocs.com/projects/fedlab-fedlab/badge/?version=latest&token=24c27118c61cc32da390946ad541028871fb336025d47404d1b6be000727ac4a)](https://fedlab-fedlab.readthedocs-hosted.com/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/SMILELab-FL/FedLab/branch/CI-management_liang/graph/badge.svg?token=4HHB5JCSC6)](https://codecov.io/gh/SMILELab-FL/FedLab)
## Introduction

FedLab是一个基于pytorch的轻量级、组件化联邦学习框架，帮助使用者在单机或多机环境下快速实现联邦学习算法的模拟。  
框架分为server和client两部分，每个角色又划分为Topology和Handler两个模块。其中，Topology模块基于torch.distributed实现，负责网络通信和消息预处理，并将处理好的信息通过预设的接口传递到底层。Handler模块负责定义优化算法，模型参数处理等工作。  

![image](/docs/imgs/fedlab-overview.png?raw=True)

### Server
server端Topology与Handler的关系如下图，Topology处理信息并调用Handler.on_receive方法，Handler消息处理逻辑和模型更新算法。  
![image](./docs/imgs/server.png?raw=True)

FedLab提供了同步联邦和异步联邦server端的demo。

### Client
client端架构和各模块功能类似于server端，但Topology和Handler的功能和处理细节有所不同。  

![image](./docs/imgs/client.png?raw=True)  

其中，异步和同步联邦的Topology通信逻辑如下图，同步联邦学习中，一轮学习的启动由server主导，而异步联邦中由client主导。  

![image](./docs/imgs/topology.png?raw=True)  

## Experiment Scene
### Standalone
![image](./docs/imgs/fedlab-standalone.png?raw=True)  
### 
### Hierarchical
![image](./docs/imgs/fedlab-hierarchical.png?raw=True)  
## Docs
文档：https://fedlab-fedlab.readthedocs-hosted.com/en/latest/

## Contribution Guidance

## Quick Start
1. 配置python环境
> pip install -r requirements.txt  
2. 运行FedAvg demo
> bash run.sh

## Citation
