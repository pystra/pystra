Installation
============

Pystra is known to run on Mac OS X, Linux and Windows, but in theory should be able to work on just about any platform for which Python, a Fortran compiler and the NumPy SciPy, and Math modules are available.

Required Dependencies
---------------------
- Python 3.8 or later
- numpy
- scipy
- matplotlib

Installation
------------
The easiest way to install `Pystra` is to use the python package index: ::

    pip install pystra

For users wishing to develop: ::

    git clone https://github.com/pystra/pystra.git
    cd pystra
    pip install -e .
    
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
Report problems with the installation, bugs in the code or feature request at the `issue tracker <http://github.com/pystra/pystra/issues>_`

