<p align="center"><img src="./docs/imgs/FedLab-logo.svg?raw=True" width=600></p>

# FedLab: A Flexible Federated Learning Framework

[![GH Actions Tests](https://github.com/SMILELab-FL/FedLab/actions/workflows/CI.yml/badge.svg)](https://github.com/SMILELab-FL/FedLab/actions) [![Documentation Status](https://readthedocs.org/projects/fedlab/badge/?version=master)](https://fedlab.readthedocs.io/en/master/?badge=master) [![License](https://img.shields.io/github/license/SMILELab-FL/FedLab)](https://opensource.org/licenses/Apache-2.0) [![codecov](https://codecov.io/gh/SMILELab-FL/FedLab/branch/v1.0/graph/badge.svg?token=4HHB5JCSC6)](https://codecov.io/gh/SMILELab-FL/FedLab) [![arXiv](https://img.shields.io/badge/arXiv-2107.11621-red.svg)](https://arxiv.org/abs/2107.11621) [![Pyversions](https://img.shields.io/pypi/pyversions/fedlab.svg?style=flat-square)](https://pypi.python.org/pypi/fedlab)


_Read this in other languages: [English](README.md), [简体中文](README.zh-cn.md)._

Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine learning, which aims to protect individual data privacy in distributed machine learning process, especially in ﬁnance, smart healthcare and edge computing. Different from traditional data-centered distributed machine learning, participants in FL setting utilize localized data to train local model, then leverages speciﬁc strategies with other participants to acquire the ﬁnal model collaboratively, avoiding direct data sharing behavior.

To relieve the burden of researchers in implementing FL algorithms and emancipate FL scientists from repetitive implementation of basic FL setting, we introduce highly customizable framework __FedLab__ in this work. __FedLab__ provides the necessary modules for FL simulation, including ***communication***, ***compression***, ***model optimization***, ***data partition*** and other ***functional modules***. Users can build FL simulation environment with custom modules like playing with LEGO bricks. For better understanding and easy usage, FL algorithm benchmark implemented in __FedLab__ are also presented.



- [Documentation](https://fedlab.readthedocs.io/en/master/)
- [Overview of FedLab](https://fedlab.readthedocs.io/en/master/overview.html)
- [Installation & Setup](https://fedlab.readthedocs.io/en/master/install.html)
- [Examples](https://fedlab.readthedocs.io/en/master/example.html)
- [Contribute Guideline](https://fedlab.readthedocs.io/en/master/contributing.html)
- [API Reference](https://fedlab.readthedocs.io/en/master/autoapi/index.html)


**The implementations of FL algorithm using FedLab can be found in** [FedLab Benchmark repository](https://github.com/SMILELab-FL/FedLab-benchmarks).

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

Contact the __FedLab__ development team through Github issues or email: 

- Dun Zeng: zengdun@foxmail.com
- Siqi Liang: zszxlsq@gmail.com

