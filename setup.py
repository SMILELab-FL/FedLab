from setuptools import setup, find_packages
from codecs import open
from os import path
from os import path as os_path
import fedlab

this_directory = os_path.abspath(os_path.dirname(__file__))


setup(
    name="fedlab",
    version=fedlab.__version__,
    keywords=["federated learning", "deep learning", "pytorch"],
    author="Dun Zeng, Siqi Liang, Xiangjing Hu",
    author_email=
    "zengdun.cs@gmail.com, zszxlsq@gmail.com, starryhu@foxmail.com",
    maintainer="Dun Zeng, Siqi Liang, Xiangjing Hu",
    maintainer_email=
    "zengdun.cs@gmail.com, zszxlsq@gmail.com, starryhu@foxmail.com",
    description=
    "A flexible Federated Learning Framework based on PyTorch, simplifying your Federated Learning research.",
    license="Apache-2.0 License",
    url="https://github.com/SMILELab-FL/FedLab",
    packages=find_packages(include=['fedlab','fedlab.*','LICENSE']),
    install_requires=['torch>=1.7.1',
                      'torchvision>=0.8.2',
                      'numpy',
                      'pandas',
                      'pynvml'],
    python_requires='>=3.6',
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    test_suite="tests.get_tests")
