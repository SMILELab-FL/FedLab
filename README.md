<p align="center"><img src="./docs/imgs/FedLab-logo.svg?raw=True" width=700></p>

# FedLab: A Flexible Federated Learning Framework

[![GH Actions Tests](https://github.com/SMILELab-FL/FedLab/actions/workflows/CI.yml/badge.svg)](https://github.com/SMILELab-FL/FedLab/actions) [![Documentation Status](https://readthedocs.org/projects/fedlab/badge/?version=master)](https://fedlab.readthedocs.io/en/master/?badge=master) [![License](https://img.shields.io/github/license/SMILELab-FL/FedLab)](https://opensource.org/licenses/Apache-2.0) [![codecov](https://codecov.io/gh/SMILELab-FL/FedLab/branch/master/graph/badge.svg?token=4HHB5JCSC6)](https://codecov.io/gh/SMILELab-FL/FedLab) [![arXiv](https://img.shields.io/badge/arXiv-2107.11621-red.svg)](https://arxiv.org/abs/2107.11621) [![Pyversions](https://img.shields.io/pypi/pyversions/fedlab.svg?style=flat-square)](https://pypi.python.org/pypi/fedlab)


Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine learning, which aims to protect individual data privacy in distributed machine learning process, especially in ﬁnance, smart healthcare and edge computing. Different from traditional data-centered distributed machine learning, participants in FL setting utilize localized data to train local model, then leverages speciﬁc strategies with other participants to acquire the ﬁnal model collaboratively, avoiding direct data sharing behavior.

To relieve the burden of researchers in implementing FL algorithms and emancipate FL scientists from repetitive implementation of basic FL setting, we introduce highly customizable framework __FedLab__ in this work. __FedLab__ provides the necessary modules for FL simulation, including ***communication***, ***compression***, ***model optimization***, ***data partition*** and other ***functional modules***. Users can build FL simulation environment with custom modules like playing with LEGO bricks. For better understanding and easy usage, FL algorithm benchmark implemented in __FedLab__ are also presented.

## Content of Documentations

- [Documentation website](https://fedlab.readthedocs.io/en/master/)
- [Overview of FedLab](https://fedlab.readthedocs.io/en/master/overview.html)
- [Installation & Setup](https://fedlab.readthedocs.io/en/master/install.html)
- [Examples](https://fedlab.readthedocs.io/en/master/example.html)
- [Contribute Guideline](https://fedlab.readthedocs.io/en/master/contributing.html)
- [API Reference](https://fedlab.readthedocs.io/en/master/autoapi/index.html)


## Quick start

1. Please read our [tutorials](./tutorials/) in jupyter notebook.

2. Run our quick start examples of different scenarios with partitioned MNIST dataset.

```
# example of standalone
$ cd ./examples/standalone/
$ bash python standalone.py --total_client 100 --com_round 3 --sample_ratio 0.1 --batch_size 100 --epochs 5 --lr 0.02
```

## Architecture
Files architecture of FedLab. These content may be helpful for users to understand our repo.

```
├── fedlab
│   ├── contrib
│   ├── core
│   ├── models
│   └── utils
├── datasets
│   └──...
├── examples
│   ├── asynchronous-cross-process-mnist
│   ├── cross-process-mnist
│   ├── hierarchical-hybrid-mnist
│   ├── network-connection-checker
│   ├── scale-mnist
│   └── standalone-mnist
└── tutorials
    ├── communication_tutorial.ipynb
    ├── customize_tutorial.ipynb
    └── pipeline_tutorial.ipynb
```

## Baselines

We provide the reproduction of baseline federated algorthms for users in this repo.

| Method              | Type   | Paper                                                        | Publication  | Official code                                        |
| ------------------- | ------ | ------------------------------------------------------------ | ------------ | ---------------------------------------------------- |
| FedAvg              | Optim. | [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) | AISTATS'2017 |                                                      |
| FedProx             | Optim. | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) | MLSys' 2020  | [Code](https://github.com/litian96/FedProx)          |
| FedDyn              | Optim. | [Federated Learning Based on Dynamic Regularization](https://openreview.net/forum?id=B7v4QMR6Z9w) | ICLR' 2021   | [Code](https://github.com/alpemreacar/FedDyn)        |
| q-FFL               | Optim. | [Fair Resource Allocation in Federated Learning](https://arxiv.org/abs/1905.10497) | ICLR' 2020   | [Code](https://github.com/litian96/fair_flearn)      |
| FedNova             | Optim. | [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) | NeurIPS'2020 | [Code](https://github.com/JYWa/FedNova)              |
| IFCA                | Optim. | [An Efficient Framework for Clustered Federated Learning](https://proceedings.neurips.cc/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html) | NeurIPS'2020 | [Code](https://github.com/jichan3751/ifca)           |
| Ditto               | Optim. | [Ditto: Fair and Robust Federated Learning Through Personalization]() | ICML'2021    | [Code](https://github.com/litian96/ditto)            |
| Power-of-choice     |  Misc. | [Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies](https://arxiv.org/abs/2010.01243) | Pre-print    |                                                      |
| SCAFFOLD            | Optim. | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning]() | ICML'2020    ||
| Personalized-FedAvg | Optim. | [Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/pdf/1909.12488.pdf) |    Pre-print      |                                                      |
| QSGD                | Com.   | [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](https://proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html) | NeurIPS'2017 |                                                      |
| NIID-Bench          | Data.  | [Federated Learning on Non-IID Data Silos: An Experimental Study](https://arxiv.org/abs/2102.02079) | ICDE' 2022 | [Code](https://github.com/Xtra-Computing/NIID-Bench) |
| LEAF                | Data.  | [LEAF: A Benchmark for Federated Settings](http://arxiv.org/abs/1812.01097) | Pre-print    | [Code](https://github.com/TalwalkarLab/leaf/)        |

## Datasets & Data Partition



## Performance & Insights

We provide the performance report of several reproduced federated learning algorithms to illustrate the correctness of FedLab in simulation. Furthermore, we describe several insights FedLab could provide for federated learning research. Without loss of generality, this section's experiments are conducted on partitioned mnist datasets. The conclusions and observations in this section should still be valid in other data sets and scenarios.

### Non-IID

We choose $\alpha = [0.1, 0.3, 0.5, 0.7]$ in label Dirichlet partitioned mnist with 100 clients. We run 200 rounds of FedAvg with 5 local batchs with full batch, learning rate 0.1 and sample ratio 0.1 (10 clients for each FL round). The test accuracy over communication round is shown below. The results reveal the most vital challenge in federated learning. 

<img src="./examples/imgs/non_iid_impacts_on_fedavg.jpg" height="400">

We provide the performance report of current FL optimization algorithms in 100 rounds.

| Algorithm      | FedAvg | FedProx | Scaffold | FedDyn | FedNova |
| -------------- | ------ | ------- | -------- | ------ | ------- |
| $\alpha = 0.1$ |        |         |          |        |         |

### Communication compression

We provide a few performance baseline in communication-efficient federated learning, which includes QSGD and top-k.

| Setting              | Baseline | QSGD-4bit | QSGD-8bit | QSGD-16bit | top-5% | Top-10% |
| -------------------- | -------- | --------- | --------- | ---------- | ------ | ------- |
| Accuracy             |          |           |           |            |        |         |
| Communication               |          |           |           |            |        |         |

## Citation

Please cite __FedLab__ in your publications if it helps your research:

```bibtex
@article{smile2021fedlab,  
    title={FedLab: A Flexible Federated Learning Framework},  
    author={Dun Zeng, Siqi Liang, Xiangjing Hu, Hui Wang and Zenglin Xu},  
    journal={arXiv preprint arXiv:2107.11621},  
    year={2021}
}
```

## Contact

Project Investigator: [Prof. Zenglin Xu](https://scholar.google.com/citations?user=gF0H9nEAAAAJ&hl=en) (xuzenglin@hit.edu.cn).

For technical issues reated to __FedLab__ development, please contact our development team through Github issues or email:

- Dun Zeng: zengdun@foxmail.com
- [Siqi Liang](https://scholar.google.com/citations?user=LIjv5BsAAAAJ&hl=en): zszxlsq@gmail.com

