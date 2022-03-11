<p align="center"><img src="./docs/imgs/FedLab-logo.svg?raw=True" width=600></p>

# FedLab: A Flexible Federated Learning Framework

[![GH Actions Tests](https://github.com/SMILELab-FL/FedLab/actions/workflows/CI.yml/badge.svg)](https://github.com/SMILELab-FL/FedLab/actions) [![Documentation Status](https://readthedocs.org/projects/fedlab/badge/?version=master)](https://fedlab.readthedocs.io/en/master/?badge=master) [![License](https://img.shields.io/github/license/SMILELab-FL/FedLab)](https://opensource.org/licenses/Apache-2.0) [![codecov](https://codecov.io/gh/SMILELab-FL/FedLab/branch/master/graph/badge.svg?token=4HHB5JCSC6)](https://codecov.io/gh/SMILELab-FL/FedLab) [![arXiv](https://img.shields.io/badge/arXiv-2107.11621-red.svg)](https://arxiv.org/abs/2107.11621) [![Pyversions](https://img.shields.io/pypi/pyversions/fedlab.svg?style=flat-square)](https://pypi.python.org/pypi/fedlab)


_Read this in other languages: [English](README.md), [简体中文](README.zh-cn.md)._

Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine learning, which aims to protect individual data privacy in distributed machine learning process, especially in ﬁnance, smart healthcare and edge computing. Different from traditional data-centered distributed machine learning, participants in FL setting utilize localized data to train local model, then leverages speciﬁc strategies with other participants to acquire the ﬁnal model collaboratively, avoiding direct data sharing behavior.

To relieve the burden of researchers in implementing FL algorithms and emancipate FL scientists from repetitive implementation of basic FL setting, we introduce highly customizable framework __FedLab__ in this work. __FedLab__ provides the necessary modules for FL simulation, including ***communication***, ***compression***, ***model optimization***, ***data partition*** and other ***functional modules***. Users can build FL simulation environment with custom modules like playing with LEGO bricks. For better understanding and easy usage, FL algorithm benchmark implemented in __FedLab__ are also presented.



- Documentation [English version](https://fedlab.readthedocs.io/en/master/)｜[中文版](https://fedlab.readthedocs.io/zh_CN/latest/)
- [Overview of FedLab](https://fedlab.readthedocs.io/en/master/overview.html)
- [Installation & Setup](https://fedlab.readthedocs.io/en/master/install.html)
- [Examples](https://fedlab.readthedocs.io/en/master/example.html)
- [Contribute Guideline](https://fedlab.readthedocs.io/en/master/contributing.html)
- [API Reference](https://fedlab.readthedocs.io/en/master/autoapi/index.html)

## Benchmarks

1. Optimization Algorithms
- [x] FedAvg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
- [x] FedAsync: [Asynchronous Federated Optimization](http://arxiv.org/abs/1903.03934)
- [x] FedProx: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
- [x] FedDyn: [Federated Learning based on Dynamic Regularization](https://openreview.net/pdf?id=B7v4QMR6Z9w)

2. Compression Algorithms
- [x] DGC: [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)
- [x] QSGD: [Communication-Efficient SGD via Gradient Quantization and Encoding](https://proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html)

3. Datasets
- [x] LEAF: [A Benchmark for Federated Settings](http://arxiv.org/abs/1812.01097)
- [x] NIID-Bench: [Federated Learning on Non-IID Data Silos: An Experimental Study](https://arxiv.org/abs/2102.02079)

 More FedLab version of FL algorithms are coming soon. For more information, please star our [FedLab Benchmark repository](https://github.com/SMILELab-FL/FedLab-benchmarks).



## Awesomes

### Tutorials

- [NeurIPS 2020] Federated Learning Tutorial [[Web]](https://sites.google.com/view/fl-tutorial/) [[Slides]](https://drive.google.com/file/d/1QGY2Zytp9XRSu95fX2lCld8DwfEdcHCG/view) [[Video]](https://slideslive.com/38935813/federated-learning-tutorial)



### Survey

- [ICLR-DPML 2021] FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks [[Paper]](https://arxiv.org/abs/2104.07145) [[Code]](https://github.com/FedML-AI/FedGraphNN)
- [arXiv 2021] Federated Graph Learning -- A Position Paper [[Paper]](https://arxiv.org/abs/2105.11099)
- [IEEE TKDE 2021] A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection [[Paper]](https://arxiv.org/pdf/1907.09693.pdf?ref=https://githubhelp.com)
- [arXiv 2021] A Survey of Fairness-Aware Federated Learning [[Paper]](https://arxiv.org/abs/2111.01872)
- [Foundations and Trends in Machine Learning 2021] Advances and Open Problems in Federated Learning [[Paper]](https://arxiv.org/abs/1912.04977)
- [arXiv 2020] Towards Utilizing Unlabeled Data in Federated Learning: A Survey and Prospective [[Paper]](https://arxiv.org/abs/2002.11545)
- [IEEE Signal Processing Magazine 2020] Federated Learning: Challenges, Methods, and Future Directions [[Paper]](https://arxiv.org/abs/1908.07873)
- [IEEE Communications Surveys & Tutorials 2020] Federated Learning in Mobile Edge Networks A Comprehensive Survey [[Paper]](https://arxiv.org/abs/1909.11875)
- [IEEE TIST 2019] Federated Machine Learning: Concept and Applications [[Paper]](https://arxiv.org/pdf/1902.04885.pdf)



### Frameworks

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



### Benchmarks

- __FedLab-benchmarks:__ [Code](https://github.com/SMILELab-FL/FedLab-benchmarks)
- [ACM TIST 2022] The OARF Benchmark Suite: Characterization and Implications for Federated Learning Systems [[Code]](https://github.com/Xtra-Computing/OARF) [[Paper]](https://arxiv.org/abs/2006.07856)
- [IEEE ICDE 2022] Federated Learning on Non-IID Data Silos: An Experimental Study [[Paper]](https://arxiv.org/abs/2102.02079) [[Official Code]](https://github.com/Xtra-Computing/NIID-Bench) [[FedLab Tutorial]](https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html)
- [ICLR-DPML 2021] FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks [[Paper]](https://arxiv.org/abs/2104.07145) [[Code]](https://github.com/FedML-AI/FedGraphNN)
- [arXiv 2018] LEAF: A Benchmark for Federated Settings [[Homepage]](https://leaf.cmu.edu/) [[Official tensorflow]](https://github.com/TalwalkarLab/leaf) [[Unofficial PyTorch]](https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/leaf) [[Paper]](https://arxiv.org/abs/1812.01097)





### FL + Semi-supervised Learning

- [ICLR 2021] Federated Semi-supervised Learning with Inter-Client Consistency & Disjoint Learning [[Paper]](https://arxiv.org/abs/2006.12097) [[Code]](https://github.com/wyjeong/FedMatch)
- [arXiv 2021] SemiFL: Communication Efficient Semi-Supervised Federated Learning with Unlabeled Clients [[Paper]](https://arxiv.org/abs/2106.01432)
- [IEEE BigData 2021] Improving Semi-supervised Federated Learning by Reducing the Gradient Diversity of Models [[Paper]](https://ieeexplore.ieee.org/abstract/document/9671693)
- [arXiv 2020] Benchmarking Semi-supervised Federated Learning [[Paper]](https://www.researchgate.net/profile/Yujun-Yan/publication/343903563_Benchmarking_Semi-supervised_Federated_Learning/links/5f571cb8299bf13a31aaff33/Benchmarking-Semi-supervised-Federated-Learning.pdf)] [[Code]](https://github.com/jhcknzzm/SSFL-Benchmarking-Semi-supervised-Federated-Learning)



### FL + HPC

- [arXiv 2022] Sky Computing: Accelerating Geo-distributed Computing in Federated Learning [[Paper]](https://arxiv.org/abs/2202.11836) [[Code]](https://github.com/hpcaitech/SkyComputing) 
- [ACM HPDC 2020] TiFL: A Tier-based Federated Learning System [[Paper]](https://arxiv.org/abs/2001.09249) [[Video]](https://www.youtube.com/watch?v=y8GZKn2zyNk)



### Awesome List

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

