# -*- coding: utf-8 -*-
# @Time    : 3/19/21 4:13 PM
# @Author  : Siqi Liang
# @Contact : zszxlsq@gmail.com
# @File    : setup.py
# @Software: PyCharm
from setuptools import setup, find_packages
from codecs import open
from os import path


def get_readme():
    here = path.abspath(path.dirname(__file__))

    # Get the long description from the README file
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


setup(
    name="FedLab",
    version="0.1",
    author="Dun Zeng, Siqi Liang",
    author_email="zengdun.cs@gmail.com, zszxlsq@gmail.com",
    description="A framework for simulation in federated setting implemented in PyTorch",
    long_description=get_readme(),
    url="https://github.com/Zengdun-cs/FedLab",
    packages=find_packages(exclude=['docs']),  # TODO: add things in 'exclude'
    install_requires=['torch',
                      'torchvision',
                      'numpy'],
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Research',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: MIT License',  # TODO: is MIT proper???

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    # test_suite="tests.get_tests"
)