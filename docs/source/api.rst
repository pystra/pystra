API Reference
=============

This section documents the full public API of Pystra.  Every class,
function, and attribute listed below is generated automatically from
the source-code docstrings using Sphinx ``autosummary``.

Core Framework
--------------

.. toctree::
    :maxdepth: 1

    system
    strong_maximum

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.model
    pystra.analysis
    pystra.system
    pystra.system_form

Reliability Methods
-------------------

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.form
    pystra.strong_maximum
    pystra.sorm
    pystra.mc
    pystra.ls
    pystra.ss
    pystra.sensitivity

Probability Transformation
--------------------------

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.transformation
    pystra.correlation
    pystra.integration
    pystra.quadrature

Load Combinations & Calibration
-------------------------------

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.fbc
    pystra.loadcomb
    pystra.calibration

Design Decision Optimization
----------------------------

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.ddo

Distributions
-------------

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.distributions
