.. _required-dependencies:

Installation
============

These instructions install **PySTRA** |release|, the version these pages
describe. Python 3.12 or later is required for the package; Python 3.13 is used
for the documentation tools.

Create an environment
----------------------

With conda, create and activate an environment::

    conda create -n pystra2.0 python=3.13
    conda activate pystra2.0

Alternatively, create a Python virtual environment with
``python -m venv .venv`` and activate it using
``source .venv/bin/activate`` on Linux/macOS or
``.venv\Scripts\Activate.ps1`` in Windows PowerShell.

Install PySTRA
--------------

Install the release from PyPI into the active environment::

    python -m pip install pystra
    python -c "import pystra; print(pystra.__version__)"

The version should be |release|.
The core installation includes NumPy, SciPy, Matplotlib and pandas.
You can now run :doc:`notebooks/ex_first_analysis`.

Optional active-learning dependencies
--------------------------------------

The Kriging and PC-Kriging examples additionally require scikit-learn, which
the ``al`` extra installs::

    python -m pip install "pystra[al]"

PCE and the classical reliability algorithms use the core installation.
Each example states its dependency requirements. Notebook downloads also
need Jupyter; install ``jupyterlab`` to run them interactively.

.. _tests:

For building all documentation and contributing, follow :doc:`developer`.
It covers the documentation extras, Pandoc and validation commands.

Install from source
-------------------

To use the latest development version, clone the repository and install it in
editable mode, with the extras you need::

    git clone --branch v2.0 https://github.com/pystra/pystra.git
    cd pystra
    python -m pip install -e ".[al]"

Using PySTRA 1.x
----------------

To keep the 1.x API, install a 1.x release and follow the
`1.x documentation <https://pystra.github.io/pystra/>`_::

    python -m pip install "pystra<2"

The two versions use different APIs. :doc:`migrating` explains how to update
existing code.

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
