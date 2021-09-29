.. _contributing:


Contributing to FedLab
========================


Reporting bugs
^^^^^^^^^^^^^^^

We use GitHub issues to track all bugs and feature requests. Feel free to open an issue if you have found a bug or wish to see a feature implemented.

In case you experience issues using this package, do not hesitate to submit a ticket to the `Bug Tracker <https://github.com/SMILELab-FL/FedLab/issues>`_. You are also welcome to post feature requests or pull requests.


Contributing Code
^^^^^^^^^^^^^^^^^^^^
You're welcome to contribute to this project through **Pull Request**. By contributing, you agree that your contributions will be licensed under `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0.html>`_ 

We encourage you to contribute to the improvement of FedLab or the FedLab implementation of existing FL methods. The preferred workflow for contributing to FedLab is to fork the main repository on GitHub, clone, and develop on a branch. Steps as follow:

1. Fork the project repository by clicking on the 'Fork'. For contributing new features, please fork FedLab `core repo <https://github.com/SMILELab-FL/FedLab>`_ or new implementations for FedLab `benchmarks repo <https://github.com/SMILELab-FL/FedLab-benchmarks>`_.

2. Clone your fork of repo from your GitHub to your local:

    .. code-block:: shell-session

        $ git clone git@github.com:YourLogin/FedLab.git
        $ cd FedLab

3. Create a new branch to save your changes:

    .. code-block:: shell-session

        $ git checkout -b my-feature

4. Develop the feature on your branch.

    .. code-block:: shell-session

        $ git add modified_files
        $ git commit

Pull Request Checklist
^^^^^^^^^^^^^^^^^^^^^^

- Please follow the file structure below for new features or create new file if there are something new.

    .. code-block:: shell-session

        fedlab
            ├── core 
            │   ├── communicator            # communication module of FedLab 
            │   ├── client                  # client related implementations
            │   │   └── scale               # scale manager and serial trainer
            │   └── server                  # server related implementations
            │       └── scale               # scale manager
            │           └── hierarchical    # hierarchical communication pattern modules
            └── utils                       # functional modules
                ├── compressor              # compressor modules
                └── dataset                 # functional modules associated with dataset

- The code should provide test cases using `unittest.TestCase`. And ensure all local tests passed:

    .. code-block:: shell-session

        $ python test_bench.py

- All public methods should have informative docstrings with sample usage presented as doctests when appropriate. Docstring and code should follow Google Python Style Guide: `中文版 <https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/>`_ | `English <https://google.github.io/styleguide/pyguide.html>`_.
