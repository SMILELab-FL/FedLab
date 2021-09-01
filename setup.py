from setuptools import setup, find_packages
from codecs import open
from os import path
from os import path as os_path
import fedlab

this_directory = os_path.abspath(os_path.dirname(__file__))


def get_readme():
    here = path.abspath(path.dirname(__file__))

    # Get the long description from the README file
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def read_requirements(filename):
    return [
        line.strip() for line in read_file(filename).splitlines()
        if not line.startswith('#')
    ]


setup(
    name="fedlab",
    version=fedlab.__version__,
    keywords=["federated learning", "depp learning", "pytorch"],
    author="Dun Zeng, Siqi Liang, Xiangjing Hu",
    author_email=
    "zengdun.cs@gmail.com, zszxlsq@gmail.com, starryhu@foxmail.com",
    maintainer="Dun Zeng, Siqi Liang, Xiangjing Hu",
    maintainer_email=
    "zengdun.cs@gmail.com, zszxlsq@gmail.com, starryhu@foxmail.com",
    description=
    "A flexible Federated Learning Framework based on PyTorch, simplifying your Federated Learning research.",
    long_description=read_file('README.md'),
    license="Apache-2.0 License",
    url="https://github.com/SMILELab-FL/FedLab",
    #packages=find_packages(exclude=['fedlab_benchmarks', 'tests', 'docs']),
    packages=find_packages(include=['fedlab','fedlab.*','LICENSE']),
    install_requires=read_requirements('requirements.txt'),
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    test_suite="tests.get_tests")
