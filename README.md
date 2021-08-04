# FedLab: A Flexible Federated Learning Framework

[![Documentation Status](https://readthedocs.com/projects/fedlab-fedlab/badge/?version=latest&token=24c27118c61cc32da390946ad541028871fb336025d47404d1b6be000727ac4a)](https://fedlab-fedlab.readthedocs-hosted.com/en/latest/?badge=latest)

_Read this in other languages: [English](README.md), [简体中文](README.zh-cn.md)._

## Table of Contents

- [Introduction](#introdcution)
- [Framework Overview](#framework-overview)
  - [Server](#server)
  - [Client](#client)
  - [Communication](#communication)
- [Experiment Scene](#experiment-scene)
  - [Standalone](#standalone)
  - [Cross-Machine](#cross-machine)
  - [Hierarchical](#hierarchical)
- [How to Use?](#how-to-use)
  - [Quick Start](#quick-start)
  - [Document](#document)
- [Citation](#citation)
- [Contribution Guideline](#contribution-guideline)
- [Contact](#contact)

## Introduction

Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine learning, which aims to protect individual data privacy in distributed machine learning process, especially in ﬁnance, smart healthcare and edge computing. Different from traditional data-centered distributed machine learning, participants in FL setting utilize localized data to train local model, then leverages speciﬁc strategies with other participants to acquire the ﬁnal model collaboratively, avoiding direct data sharing behavior.

To relieve the burden of researchers in implementing FL algorithms and emancipate FL scientists from repetitive implementation of basic FL setting, we introduce highly customizable framework __FedLab__ in this work. __FedLab__ provides the necessary modules for FL simulation, including communication, compression, model optimization, data partition and other functional modules. __FedLab__ users can build FL simulation environment with custom modules like playing with LEGO bricks. For better understanding and easy usage, FL algorithm benchmark implemented in __FedLab__ are also presented.

For more details, please read our [full paper](https://arxiv.org/abs/2107.11621).



## Framework Overview

<object data="./docs/imgs/fedlab-overview.pdf?raw=True" type="application/pdf" width="700px" height="700px">
    <embed src="./docs/imgs/fedlab-overview.pdf?raw=True">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="./docs/imgs/fedlab-overview.pdf?raw=True">Download PDF</a>.</p>
    </embed>
</object>

![framework-overview](./docs/imgs/fedlab-overview.pdf?raw=True)

__FedLab__ provides two basic roles in FL setting: `Server` and `Client`. Each `Server`/`Client` consists of two components called `NetworkManager` and `ParameterHandler`/`Trainer`. 

- `NetworkManager` module manages message process task, which provides interfaces to customize communication agreements and compression
- `ParameterHandler` is responsible for backend computation in `Server`; and `Trainer` is in charge of backend computation in `Client` 




### Server

The connection between `NetworkManager` and `ParameterServerHandler` in `Server` is shown as below. `NetworkManager` processes message and calls `ParameterServerHandler.on_receive()` method, while `ParameterServerHandler` performs training as well as computation process of server (model aggregation for example), and updates the global model. 

<img src="./docs/imgs/fedlab-server.pdf?raw=True" width=450>

### Client

`Client` shares similar design and structure with `Server`, with `NetworkManager` in charge of message processing as well as network communication with server, and `Trainer` for client local training procedure.

<img src="./docs/imgs/fedlab-client.pdf?raw=True" width=450>



### Communication

__FedLab__ furnishes both synchronous and asynchronous communication patterns, and their corresponding communication logics of `NetworkManager` is shown as below.

1. Synchronous FL: each round is launched by server, that is, server performs clients sampling first then broadcasts global model parameters.

![syn-comm](./docs/imgs/fedlab-sychronous.pdf?raw=True)

2. Asynchronous FL: each round is launched by clients, that is, clients request current global model parameters then perform local training.

![asyn-comm](./docs/imgs/fedlab-asychronous.pdf?raw=True)

## Experiment Scene

__FedLab__ supports both single machine and  multi-machine FL simulations, with _standalone_ mode for single machine experiments, while _corss-machine_ mode and _hierarchical_ mode for multi-machine experiments.

### Standalone

__FedLab__ implements `SerialTrainer` for FL simulation in single system process. `SerialTrainer` allows user to simulate a FL system with multiple clients executing one by one in serial in one `SerialTrainer`. It is designed for simulation in environment with limited computation resources.  
![image](./docs/imgs/fedlab-SerialTrainer.pdf?raw=True)

### Cross-Machine

__FedLab__ supports simulation executed on multiple machines with correct network topology conﬁguration. More ﬂexibly in parallel, `SerialTrainer` is able to replace the regular `Trainer`. In this way, machine with more computation resources can be assigned with more workload of simulating. 

> All machines must be in the same network (LAN or WAN) for cross-machine deployment.

![image](./docs/imgs/fedlab-multi_process.pdf?raw=True)

### Hierarchical

_Hierarchical_ mode for __FedLab__ is designed for situations where both _standalone_ and _cross-machine_ are insufficient for simulation. __FedLab__ promotes `Scheduler` as middle-server to organize client groups. Each `Scheduler` manages the communication between server and a client group containing a subset of clients. And server can communicate with clients in different LAN via corresponding `Scheduler`. 

> The client group for each schedular can be either _standalone_ or _cross-machine_.

A hierarchical FL system with $K$​​ client groups is depicted as below.
![image](./docs/imgs/fedlab-hierarchical.pdf?raw=True)

## How to use？

### Quick Start

1. Install package dependence:

```shell
pip install -r requirements.txt
```

2. Enter `./fedlab_benchmarks/algorithm/fedavg/`, run FedAvg demo:

```shell
# cd into fedlab_benchmarks/algorithm/fedavg/ directory
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

- Dun Zeng: zengdun@foxmail.com
- Siqi Liang: zszxlsq@gmail.com

