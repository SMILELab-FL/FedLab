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





## 资源

### 教程

- [NeurIPS 2020] Federated Learning Tutorial [[Web]](https://sites.google.com/view/fl-tutorial/) [[Slides]](https://drive.google.com/file/d/1QGY2Zytp9XRSu95fX2lCld8DwfEdcHCG/view) [[Video]](https://slideslive.com/38935813/federated-learning-tutorial)



### 概述

- [ICLR-DPML 2021] FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks [[Paper]](https://arxiv.org/abs/2104.07145) [[Code]](https://github.com/FedML-AI/FedGraphNN)
- [arXiv 2021] Federated Graph Learning -- A Position Paper [[Paper]](https://arxiv.org/abs/2105.11099)
- [IEEE TKDE 2021] A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection [[Paper]](https://arxiv.org/pdf/1907.09693.pdf?ref=https://githubhelp.com)
- [arXiv 2021] A Survey of Fairness-Aware Federated Learning [[Paper]](https://arxiv.org/abs/2111.01872)
- [Foundations and Trends in Machine Learning 2021] Advances and Open Problems in Federated Learning [[Paper]](https://arxiv.org/abs/1912.04977)
- [arXiv 2020] Towards Utilizing Unlabeled Data in Federated Learning: A Survey and Prospective [[Paper]](https://arxiv.org/abs/2002.11545)
- [IEEE Signal Processing Magazine 2020] Federated Learning: Challenges, Methods, and Future Directions [[Paper]](https://arxiv.org/abs/1908.07873)
- [IEEE Communications Surveys & Tutorials 2020] Federated Learning in Mobile Edge Networks A Comprehensive Survey [[Paper]](https://arxiv.org/abs/1909.11875)
- [IEEE TIST 2019] Federated Machine Learning: Concept and Applications [[Paper]](https://arxiv.org/pdf/1902.04885.pdf)



### 框架

- __FedLab:__ [Code](https://github.com/SMILELab-FL/FedLab), [FedLab-benchmarks](https://github.com/SMILELab-FL/FedLab-benchmarks), [Doc](https://fedlab.readthedocs.io/) ([zh-CN-Doc](https://fedlab.readthedocs.io/zh_CN/latest/)), [Paper](https://arxiv.org/abs/2107.11621)
- __Flower:__ [Code](https://github.com/adap/flower), [Homepage](https://flower.dev/), [Doc](https://flower.dev/docs/), [Paper](https://arxiv.org/abs/2007.14390)
- __FedML:__ [Code](https://github.com/FedML-AI/FedML), [Doc](http://doc.fedml.ai/#/), [Paper](https://arxiv.org/abs/2007.13518)
- __FedLearn:__ [Code](https://github.com/cyqclark/fedlearn-algo), [Paper](https://arxiv.org/abs/2107.04129)
- __PySyft:__ [Code](https://github.com/OpenMined/PySyft), [Doc](https://pysyft.readthedocs.io/en/latest/installing.html), [Paper](https://arxiv.org/abs/1811.04017)
- __TensorFlow Federated (TFF):__ [Code](https://github.com/tensorflow/federated), [Doc](https://www.tensorflow.org/federated)
- __FEDn:__ [Code](https://github.com/scaleoutsystems/fedn), [Paper](https://arxiv.org/abs/2103.00148)
- __FATE:__ [Code](https://github.com/FederatedAI/FATE), [Homepage](https://www.fedai.org/), [Doc](https://fate.readthedocs.io/en/latest/), [Paper](https://www.jmlr.org/papers/v22/20-815.html)
- __PaddleFL:__ [Code](https://github.com/PaddlePaddle/PaddleFL), [Doc](https://paddlefl.readthedocs.io/en/latest/index.html)
- __Fedlearner:__ [Code](https://github.com/bytedance/fedlearner)
- __OpenFL:__ [Code](https://github.com/intel/openfl), [Doc](https://openfl.readthedocs.io/en/latest/install.html), [Paper](https://arxiv.org/abs/2105.06413)



### 基准

- __FedLab-benchmarks:__ [Code](https://github.com/SMILELab-FL/FedLab-benchmarks)
- [ACM TIST 2022] The OARF Benchmark Suite: Characterization and Implications for Federated Learning Systems [[Code]](https://github.com/Xtra-Computing/OARF) [[Paper]](https://arxiv.org/abs/2006.07856)
- [IEEE ICDE 2022] Federated Learning on Non-IID Data Silos: An Experimental Study [[Paper]](https://arxiv.org/abs/2102.02079) [[Official Code]](https://github.com/Xtra-Computing/NIID-Bench) [[FedLab Tutorial]](https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html)
- [ICLR-DPML 2021] FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks [[Paper]](https://arxiv.org/abs/2104.07145) [[Code]](https://github.com/FedML-AI/FedGraphNN)
- [arXiv 2018] LEAF: A Benchmark for Federated Settings [[Homepage]](https://leaf.cmu.edu/) [[Official tensorflow]](https://github.com/TalwalkarLab/leaf) [[Unofficial PyTorch]](https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/leaf) [[Paper]](https://arxiv.org/abs/1812.01097)





### 联邦 + 半监督

- [ICLR 2021] Federated Semi-supervised Learning with Inter-Client Consistency & Disjoint Learning [[Paper]](https://arxiv.org/abs/2006.12097) [[Code]](https://github.com/wyjeong/FedMatch)
- [arXiv 2021] SemiFL: Communication Efficient Semi-Supervised Federated Learning with Unlabeled Clients [[Paper]](https://arxiv.org/abs/2106.01432)
- [IEEE BigData 2021] Improving Semi-supervised Federated Learning by Reducing the Gradient Diversity of Models [[Paper]](https://ieeexplore.ieee.org/abstract/document/9671693)
- [arXiv 2020] Benchmarking Semi-supervised Federated Learning [[Paper]](https://www.researchgate.net/profile/Yujun-Yan/publication/343903563_Benchmarking_Semi-supervised_Federated_Learning/links/5f571cb8299bf13a31aaff33/Benchmarking-Semi-supervised-Federated-Learning.pdf)] [[Code]](https://github.com/jhcknzzm/SSFL-Benchmarking-Semi-supervised-Federated-Learning)



### 联邦 + 高性能计算

- [arXiv 2022] Sky Computing: Accelerating Geo-distributed Computing in Federated Learning [[Paper]](https://arxiv.org/abs/2202.11836) [[Code]](https://github.com/hpcaitech/SkyComputing) 
- [ACM HPDC 2020] TiFL: A Tier-based Federated Learning System [[Paper]](https://arxiv.org/abs/2001.09249) [[Video]](https://www.youtube.com/watch?v=y8GZKn2zyNk)



### 相关资源

- [chaoyanghe/Awesome-Federated-Learning](https://github.com/chaoyanghe/Awesome-Federated-Learning)
- [weimingwill/awesome-federated-learning](https://github.com/weimingwill/awesome-federated-learning)
- [tushar-semwal/awesome-federated-computing](https://github.com/tushar-semwal/awesome-federated-computing)
- [ZeroWangZY/federated-learning](https://github.com/ZeroWangZY/federated-learning)
- [innovation-cat/Awesome-Federated-Machine-Learning](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning)
- [huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers](https://github.com/huweibo/Awesome-Federated-Learning-on-Graph-and-GNN-papers)





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
- [Siqi Liang](https://scholar.google.com/citations?user=LIjv5BsAAAAJ&hl=en): zszxlsq@gmail.com





