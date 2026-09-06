Installation
============

Pystra runs on Mac OS X, Linux, and Windows — any platform where
Python 3 and the required scientific packages are available.

Required Dependencies
---------------------
- Python 3.9 or later
- numpy
- scipy 1.11 or later
- matplotlib
- pandas

Installation
------------
The easiest way to install the latest stable `Pystra` release is from PyPI: ::

    pip install pystra

To work with the 2.0 development API documented on this branch: ::

    git clone --branch v2.0 https://github.com/pystra/pystra.git
    cd pystra
    pip install -e ".[test]"

For contributions, first fork the repo and clone from your fork.
`Here <https://www.dataschool.io/how-to-contribute-on-github/>`_ is a good guide on this workflow.

Tests
-----
`Pystra` comes with ``pytest`` functions to verify the correct functioning of the package.
Users can test this using: ::

    python -m pytest

from the root directory of the package.

Bugs and feature requests
-------------------------
Report problems with the installation, bugs in the code, or feature
requests at the `issue tracker <https://github.com/pystra/pystra/issues>`_.
