.. _required-dependencies:

Installation
============

These instructions install **PySTRA** |release| **from the v2.0 branch**, matching
the API used in these pages. Python 3.9 or later is required for the package;
Python 3.13 is used for the documentation tools.

Create an environment
----------------------

With conda, create and activate an environment::

    conda create -n pystra2.0 python=3.13
    conda activate pystra2.0

Alternatively, create a Python virtual environment with
``python -m venv .venv`` and activate it using
``source .venv/bin/activate`` on Linux/macOS or
``.venv\Scripts\Activate.ps1`` in Windows PowerShell.

Install the development package
-------------------------------

Clone the development branch, then install it into the active environment::

    git clone --branch v2.0 https://github.com/pystra/pystra.git
    cd pystra
    python -m pip install -e .
    python -c "import pystra; print(pystra.__version__)"

The version should begin with ``2.0``; the current version is |release|.
The core installation includes NumPy, SciPy, Matplotlib and pandas.
You can now run :doc:`notebooks/ex_first_analysis`.

Optional active-learning dependencies
--------------------------------------

The Kriging and PC-Kriging examples additionally require scikit-learn.
From the same checkout, install::

    python -m pip install -e ".[al]"

PCE and the classical reliability algorithms use the core installation.
Each example states its dependency requirements. Notebook downloads also
need Jupyter; install ``jupyterlab`` to run them interactively.

.. _tests:

For building all documentation and contributing, follow :doc:`developer`.
It covers the documentation extras, Pandoc and validation commands.

Using the stable release
-------------------------

To use the released 1.x API, install ``pystra`` from PyPI and follow the
`stable documentation <https://pystra.github.io/pystra/>`_::

    python -m pip install pystra

The stable release and the development examples use different APIs.
:doc:`migrating` explains how to update existing code.

.. _bugs-and-feature-requests:

Troubleshooting installation
----------------------------

If a notebook reports an unexpected version or missing package, check the
interpreter used by its kernel::

    import sys
    import pystra
    print(sys.executable)
    print(pystra.__version__)

Select the kernel for the environment where PySTRA is installed. For further
help, report the interpreter, package version and error message in the
`issue tracker <https://github.com/pystra/pystra/issues>`_.
