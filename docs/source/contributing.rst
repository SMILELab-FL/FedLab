.. _contributing:


Contributing to FedLab
========================


Reporting bugs
^^^^^^^^^^^^^^^

We use GitHub issues to track all bugs and feature requests. Feel free to open an issue if you have found a bug or wish to see a feature implemented.

In case you experience issues using this package, do not hesitate to submit a ticket to the `Bug Tracker <https://github.com/SMILELab-FL/FedLab/issues>`_. You are also welcome to post feature requests or pull requests.


Guidelines
^^^^^^^^^^^^^^^^^^^^
You're welcome to contribute to this project through **Pull Request**. By contributing, you agree that your contributions will be licensed under `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0.html>`_ 

We encourage you to contribute to the improvement of FedLab or the FedLab reproduction of existing FL methods. Before your pull request, please make sure that your are familiar with the code structure:

.. code-block:: shell-session

    fedlab
    │   ├── core 
    │   │   ├── communicator            # communication module of FedLab 
    │   │   ├── client                  # client related implementations
    │   │   │   └── scale               # scale manager and serial trainer
    │   │   └── server                  # server related implementations
    │   │       └── scale               # scale manager
    │   │           └── hierarchical    # hierarchical communication pattern modules
    │   └── utils                       # functional modules
    │       └── dataset                 # functional modules associated with dataset
    │
    fedlab_benchmarks
        ├── algorithm                   # implementations of existing FL methods
        └── datasets                    # FL dataset

Coding
----------------
- Please make sure your contributions corresponds to the code structure.
- The code should provide test cases using `unittest.TestCase`.

Docstring
----------
Docstring and code should follow Google Python Style Guide: `中文版 <https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/>`_ | `English <https://google.github.io/styleguide/pyguide.html>`_.
