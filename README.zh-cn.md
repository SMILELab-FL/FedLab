<p align="center"><img src="./docs/imgs/FedLab-logo.svg?raw=True" width=600></p>

# FedLab: A Flexible Federated Learning Framework

[![GH Actions Tests](https://github.com/SMILELab-FL/FedLab/actions/workflows/CI.yml/badge.svg)](https://github.com/SMILELab-FL/FedLab/actions) [![Documentation Status](https://readthedocs.org/projects/fedlab/badge/?version=master)](https://fedlab.readthedocs.io/en/master/?badge=master) [![License](https://img.shields.io/github/license/SMILELab-FL/FedLab)](https://opensource.org/licenses/Apache-2.0) [![codecov](https://codecov.io/gh/SMILELab-FL/FedLab/branch/master/graph/badge.svg?token=4HHB5JCSC6)](https://codecov.io/gh/SMILELab-FL/FedLab) [![arXiv](https://img.shields.io/badge/arXiv-2107.11621-red.svg)](https://arxiv.org/abs/2107.11621) [![Pyversions](https://img.shields.io/pypi/pyversions/fedlab.svg?style=flat-square)](https://pypi.python.org/pypi/fedlab)


_其他语言版本：[English](README.md), [简体中文](README.zh-cn.md)._

​        由谷歌最先提出的联邦学习近来成为机器学习研究中一个迅速发展的领域。联邦学习的目标是在分布式机器学习中保护个体数据隐私，尤其是金融领域、智能医疗以及边缘计算领域。不同于传统的数据中心式的分布式机器学习，联邦学习中的参与者利用本地数据训练本地模型，然后利用具体的聚合策略结合从其他参与者学习到的知识，来合作生成最终的模型。这种学习方式避免了直接分享数据的行为。

​        为了减轻研究者实现联邦学习算法的负担，我们向大家介绍非常灵活的联邦学习框架**FedLab**。**FedLab**为联邦学习的模拟实验提供了必要的模块，包括通信、压缩、模型优化、数据切分，及其他功能性模块。用户们可以像使用乐高积木一样，根据需求构建他们的联邦模拟环境。我们还提供了一些联邦学习的基准算法的实现，方便用户能更好的理解并使用**FedLab**。



- [英文版文档](https://fedlab.readthedocs.io/en/master/) | [中文版文档](https://fedlab.readthedocs.io/zh_CN/latest/)
- [FedLab简介](https://fedlab.readthedocs.io/en/master/overview.html)
- [安装与设置](https://fedlab.readthedocs.io/en/master/install.html)
- [使用例子](https://fedlab.readthedocs.io/en/master/example.html)
- [如何贡献代码](https://fedlab.readthedocs.io/en/master/contributing.html)
- [API介绍](https://fedlab.readthedocs.io/en/master/autoapi/index.html)
- [联系方式](#联系方式)

## 基准实现

1. 优化算法
- [x] FedAvg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
- [x] FedAsync: [Asynchronous Federated Optimization](http://arxiv.org/abs/1903.03934)
- [x] FedProx: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
- [x] FedDyn: [Federated Learning based on Dynamic Regularization](https://openreview.net/pdf?id=B7v4QMR6Z9w)

2. 压缩算法
- [x] DGC: [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)
- [x] QSGD: [Communication-Efficient SGD via Gradient Quantization and Encoding](https://proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html)

3. 数据集
- [x] LEAF: [A Benchmark for Federated Settings](http://arxiv.org/abs/1812.01097)
- [x] NIID-Bench: [Federated Learning on Non-IID Data Silos: An Experimental Study](https://arxiv.org/abs/2102.02079)

更多FedLab版本的FL算法即将推出。有关更多信息，请关注我们的[FedLab基准算法库](https://github.com/SMILELab-FL/FedLab-benchmarks)。

## 如何贡献代码

欢迎提交pull request贡献代码。

- 代码应遵循[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)协议
- 代码及注释规范遵守谷歌Python风格指南：[中文版](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)|[English](https://google.github.io/styleguide/pyguide.html)
- 代码需要提供用`unittest.TestCase`编写的测试样例



## 引用

如果**FedLab**对您的研究工作有所帮助，请引用我们的论文：

```bibtex
@article{smile2021fedlab,  
    title={FedLab: A Flexible Federated Learning Framework},  
    author={Dun Zeng, Siqi Liang, Xiangjing Hu and Zenglin Xu},  
    journal={arXiv preprint arXiv:2107.11621},  
    year={2021}
}
```



## 联系方式

项目负责人：[徐增林教授](https://scholar.google.com/citations?user=gF0H9nEAAAAJ&hl=en)（xuzenglin@hit.edu.cn）。

技术细节以及开发问题，请通过GitHub issues或邮件联系**FedLab**开发团队：

- Dun Zeng: zengdun@foxmail.com
- Siqi Liang: zszxlsq@gmail.com





