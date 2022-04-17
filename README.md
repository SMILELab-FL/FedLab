<p align="center"><img src="./docs/imgs/FedLab-logo.svg?raw=True" width=600></p>

# FedLab: A Flexible Federated Learning Framework

[![GH Actions Tests](https://github.com/SMILELab-FL/FedLab/actions/workflows/CI.yml/badge.svg)](https://github.com/SMILELab-FL/FedLab/actions) [![Documentation Status](https://readthedocs.org/projects/fedlab/badge/?version=master)](https://fedlab.readthedocs.io/en/master/?badge=master) [![License](https://img.shields.io/github/license/SMILELab-FL/FedLab)](https://opensource.org/licenses/Apache-2.0) [![codecov](https://codecov.io/gh/SMILELab-FL/FedLab/branch/master/graph/badge.svg?token=4HHB5JCSC6)](https://codecov.io/gh/SMILELab-FL/FedLab) [![arXiv](https://img.shields.io/badge/arXiv-2107.11621-red.svg)](https://arxiv.org/abs/2107.11621) [![Pyversions](https://img.shields.io/pypi/pyversions/fedlab.svg?style=flat-square)](https://pypi.python.org/pypi/fedlab)


_Read this in other languages: [English](README.md), [简体中文](README.zh-cn.md)._

Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine learning, which aims to protect individual data privacy in distributed machine learning process, especially in ﬁnance, smart healthcare and edge computing. Different from traditional data-centered distributed machine learning, participants in FL setting utilize localized data to train local model, then leverages speciﬁc strategies with other participants to acquire the ﬁnal model collaboratively, avoiding direct data sharing behavior.

To relieve the burden of researchers in implementing FL algorithms and emancipate FL scientists from repetitive implementation of basic FL setting, we introduce highly customizable framework __FedLab__ in this work. __FedLab__ provides the necessary modules for FL simulation, including ***communication***, ***compression***, ***model optimization***, ***data partition*** and other ***functional modules***. Users can build FL simulation environment with custom modules like playing with LEGO bricks. For better understanding and easy usage, FL algorithm benchmark implemented in __FedLab__ are also presented.


## Documentations

- Documentation [English version](https://fedlab.readthedocs.io/en/master/)｜[中文版](https://fedlab.readthedocs.io/zh_CN/latest/)
- [Overview of FedLab](https://fedlab.readthedocs.io/en/master/overview.html)
- [Installation & Setup](https://fedlab.readthedocs.io/en/master/install.html)
- [Examples](https://fedlab.readthedocs.io/en/master/example.html)
- [Contribute Guideline](https://fedlab.readthedocs.io/en/master/contributing.html)
- [API Reference](https://fedlab.readthedocs.io/en/master/autoapi/index.html)


## [Quick start with demos](./examples/README.md)

New FedLab (v1.2.0) provides fully finished communication pattern. We futher simplified the APIs of NetworkManager part and re-organised the APIs of trainer. Currently, three basic scenes (Standalone, Cross-process and Hierachical) are supported by choosing different client trainer. Please see our [demos](./examples/README.md).

## [FedLab Benchmarks](https://github.com/SMILELab-FL/FedLab-benchmarks)

Thanks to our contributors, algorithms and benchmarks are provided in our [FedLab-Benchmarks repo](https://github.com/SMILELab-FL/FedLab-benchmarks). More FedLab version of FL algorithms are coming.

1. Optimization Algorithms
- [x] FedAvg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
- [x] FedAsync: [Asynchronous Federated Optimization](http://arxiv.org/abs/1903.03934)
- [x] FedProx: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
- [x] FedDyn: [Federated Learning based on Dynamic Regularization](https://openreview.net/pdf?id=B7v4QMR6Z9w)
- [x] Personalized-FedAvg: [Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/pdf/1909.12488.pdf)
2. Compression Algorithms
- [x] DGC: [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)
- [x] QSGD: [Communication-Efficient SGD via Gradient Quantization and Encoding](https://proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html)

3. Datasets
- [x] LEAF: [A Benchmark for Federated Settings](http://arxiv.org/abs/1812.01097)
- [x] NIID-Bench: [Federated Learning on Non-IID Data Silos: An Experimental Study](https://arxiv.org/abs/2102.02079)

## Awesome paper lists

- [NeurIPS 2020] Federated Learning Tutorial [[Web]](https://sites.google.com/view/fl-tutorial/) [[Slides]](https://drive.google.com/file/d/1QGY2Zytp9XRSu95fX2lCld8DwfEdcHCG/view) [[Video]](https://slideslive.com/38935813/federated-learning-tutorial)
- [chaoyanghe/Awesome-Federated-Learning](https://github.com/chaoyanghe/Awesome-Federated-Learning)
- [weimingwill/awesome-federated-learning](https://github.com/weimingwill/awesome-federated-learning)
- [tushar-semwal/awesome-federated-computing](https://github.com/tushar-semwal/awesome-federated-computing)
- [ZeroWangZY/federated-learning](https://github.com/ZeroWangZY/federated-learning)
- [innovation-cat/Awesome-Federated-Machine-Learning](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning)
- [huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers](https://github.com/huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers)



## Contribution

You're welcome to contribute to this project through _Pull Request_.

- By contributing, you agree that your contributions will be licensed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html) 
- Docstring  and code should follow Google Python Style Guide: [中文版](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)|[English](https://google.github.io/styleguide/pyguide.html)
- The code should provide test cases using `unittest.TestCase`



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

## Contact

Project Investigator: [Prof. Zenglin Xu](https://scholar.google.com/citations?user=gF0H9nEAAAAJ&hl=en) (xuzenglin@hit.edu.cn).

For technical issues reated to __FedLab__ development, please contact our development team through Github issues or email:

- Dun Zeng: zengdun@foxmail.com
- [Siqi Liang](https://scholar.google.com/citations?user=LIjv5BsAAAAJ&hl=en): zszxlsq@gmail.com

