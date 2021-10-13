************
Installation
************

Pystra is known to run on Mac OS X, Linux and Windows, but in theory should be
able to work on just about any platform for which Python, a Fortran compiler
and the NumPy SciPy, and Math modules are available. However, installing some
extra depencies can greatly improve Pystra's performance and versatility. The
following describes the required and optional dependencies and takes you
through the installation process.


Dependencies
============

Pystra requires some prerequisite packages to be present on the system.
Fortunately, there are currently only a few hard dependencies, and all are
freely available online.

* `Python`_ version 3.4 or later.

* `NumPy`_ : The fundamental scientific programming package, it
  provides a multidimensional array type and many useful functions for
  numerical analysis.

* `SciPy`_ : Library of algorithms for mathematics, science and engineering.

* `IPython`_ (optional): An enhanced interactive Python shell and an
  architecture for interactive parallel computing.


.. _`Python`: http://www.python.org/.

.. _`NumPy`: http://www.scipy.org/NumPy

.. _`SciPy`: http://www.scipy.org/

.. _`IPython`: http://ipython.scipy.org/



Installation using pip
======================

Pystra is pure python code. It has no platform-specific dependencies and should thus work on all platforms. It builds on `numpy` and `scipy`. The latest version of `pystra` can be installed by typing:::

   pip install git+git://github.com/pystra/pystra.git

Installation using setup.py
===========================

First download the source code from `GitHub`_ and unpack it. Then move
into the unpacked directory and run:::

  python setup.py install


Development version
===================

You can check out the development version of the code from the `GitHub`_
repository::

    git clone git://github.com/pystra/pystra.git

.. _`GitHub`: https://github.com/pystra/pystra


Bugs and feature requests
=========================

Report problems with the installation, bugs in the code or feature request at
the `issue tracker`_. 

.. _`issue tracker`: http://github.com/pystra/pystra/issues

