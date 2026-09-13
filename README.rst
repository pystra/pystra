.. figure:: https://raw.githubusercontent.com/pystra/pystra/main/docs/source/images/logo/logo_pystra_mid.png
   :alt: PySTRA logo
   :align: center
   :width: 398px

PySTRA - Python Structural Reliability Analysis
===============================================

PySTRA provides a carefully validated implementation of established and
selected modern structural reliability methods, coupled with practical tools
for code calibration and structural assessment. It integrates with NumPy,
SciPy and pandas, and supports reliability models defined by Python functions.

Installation
------------

Install PySTRA from PyPI::

   python -m pip install pystra

Use ``python -m pip install "pystra[al]"`` for the optional active-learning
methods. The `installation guide <https://pystra.github.io/pystra/2.0/install.html>`_ covers environments,
notebooks and installing from source.

PySTRA 2.0 is not compatible with 1.x. `What's new <https://pystra.github.io/pystra/2.0/whatsnew.html>`_
summarizes the release, and the `migration guide <https://pystra.github.io/pystra/2.0/migrating.html>`_ shows
how to update 1.x code; ``python -m pystra.migrate`` makes the unambiguous
changes for you. To stay on 1.x, install ``"pystra<2"`` and use the
`1.x documentation <https://pystra.github.io/pystra/>`_.

Features
--------

* FORM and SORM, direct and importance sampling, line sampling and subset simulation.
* Explicit copulas and probability transformations, component and system reliability.
* Code calibration with normalized reliability, load combinations and FBC processes.
* Structural decision studies, target reliability and consequence/cost models.
* Optional surrogate-assisted reliability with independent benchmark checks.
* Result records, tables and reusable plotting helpers for engineering studies.

Limit states are Python functions. Each algorithm has response-smoothness,
transformation and convergence requirements; use the
`method-selection guide <https://pystra.github.io/pystra/2.0/guides/methods.html>`_ to choose a starting
point and plan validation.

Getting started
---------------

The `first analysis <https://pystra.github.io/pystra/2.0/notebooks/ex_first_analysis.html>`_ checks FORM
against an exact resistance-minus-load probability. Follow the
`example categories <https://pystra.github.io/pystra/2.0/tutorial.html>`_ for more advanced methods and the
`benchmark catalogue <https://pystra.github.io/pystra/2.0/benchmarks.html>`_ for published problems. For
problems with very small failure probabilities, see the
`high-reliability guide <https://pystra.github.io/pystra/2.0/guides/high_reliability.html>`_.

Contributing
------------

See the `contributor guide <https://github.com/pystra/pystra/blob/v2.0/CONTRIBUTING.md>`_
for coding conventions, numerical validation, building the documentation and
pull-request guidance.

Credits
-------
PySTRA is built on PyRe by Jürgen Hackl; FERUM4.1 by Jean-Marc Bourinet; FERUM by Terje Haukaas and Armen Der Kiureghian.

Copyright 2021 The Pystra Developers.

List of references
------------------

[Bourinet2009] J.-M. Bourinet, C. Mattrand, and V Dubourg. A review of recent features and improvements added to FERUM software. In Proc. of the 10th International Conference on Structural Safety and Reliability (ICOSSAR’09), Osaka, Japan, 2009.

[Bourinet2010] J.-M. Bourinet. FERUM 4.1 User’s Guide, 2010.

[DerKiureghian2006] A. Der Kiureghian, T. Haukaas, and K. Fujimura. Structural reliability software at the University of California, Berkeley. Structural Safety, 28(1-2):44–67, 2006.

[Hackl2013] J. Hackl. Generic Framework for Stochastic Modeling of Reinforced Concrete Deterioration Caused by Corrosion. Master’s thesis, Norwegian University of Science and Technology, Trondheim, Norway, 2013.

License
~~~~~~~

PySTRA is distributed under the GNU General Public License, version 3 or
later (GPL-3.0-or-later). See ``LICENSE`` for the terms and retained PyRe notice.
