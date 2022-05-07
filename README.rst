.. figure:: docs/source/images/logo/logo_pystra_mid.png
   :alt: Pystra logo
   :align: center
   :scale: 50

***********************************************
Pystra - Python Structural Reliability Analysis
***********************************************

Pystra (Python Structural Reliability Analysis) is a python module for structural reliability analysis. Its flexibility and extensibility make it applicable to a large
suite of problems. Along with core reliability analysis functionality, Pystra
includes methods for summarizing output. Pystra is also closely integrated with the usual python scientific packages workflow, numpy and scipy; in particular, all statistical distributions in Scipy can be used in reliability modeling.

Installation
============

To install *Pystra* just do:

  $ pip install pystra

Features
========

Pystra provides functionalities to make structural reliability analysis as easy
as possible. Here is a short list of some of its features:

* Perform reliability analysis with different kinds of Reliability Methods.

* Perform reliability analysis with Crude Monte Carlo Simulation.

* Includes a large suite of well-documented statistical distributions.

* Uses NumPy for numerics wherever possible.

* No limitation on the limit state function.

* Correlation between the random variables are possible.

* Traces can be saved to the disk as plain text.

* Pystra can be embedded in larger programs, and results can be analyzed
  with the full power of Python.


Getting started
===============

This `Documentation`_ provides all the information needed to install Pystra, code a
reliability model, run the sampler, save and visualize the results. In
addition, it contains a list of the statistical distributions currently
available.

.. _`Documentation`: http://pystra.github.io/pystra/

.. _`FERUM`: http://www.ce.berkeley.edu/projects/ferum/

.. _`IFMA`: http://www.ifma.fr/Recherche/Labos/FERUM

Credits
=======
Pystra is built on PyRe by Jürgen Hackl; FERUM4.1 by Jean-Marc Bourinet; FERUM by Terje Haukaas and Armen Der Kiureghian.

Copyright 2021 The Pystra Developers.

List of References
==================

[Bourinet2009] J.-M. Bourinet, C. Mattrand, and V Dubourg. A review of recent features and improvements added to FERUM software. In Proc. of the 10th International Conference on Structural Safety and Reliability (ICOSSAR’09), Osaka, Japan, 2009.

[Bourinet2010] J.-M. Bourinet. FERUM 4.1 User’s Guide, 2010.

[DerKiureghian2006] A. Der Kiureghian, T. Haukaas, and K. Fujimura. Structural reliability software at the University of California, Berkeley. Structural Safety, 28(1-2):44–67, 2006.

[Hackl2013] J. Hackl. Generic Framework for Stochastic Modeling of Reinforced Concrete Deterioration Caused by Corrosion. Master’s thesis, Norwegian University of Science and Technology, Trondheim, Norway, 2013.
