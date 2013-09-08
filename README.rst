*******************************************
Structural Reliability Analysis with Python
*******************************************

:Date: September 2013
:Authors: Jürgen Hackl
:Contact: hackl.j@gmx.at
:Web site: http://github.com/hackl/pyre
:Documentation: http://hackl.github.io/pyre/
:Copyright: This document has been placed in the public domain.
:License: PyRe is released under the GNU General Public Licence.
:Version: 5.0.2


Note
----

   If you have any problems, found bugs in the code or have feature request
   comments or questions, please feel free to send a mail to `Jürgen Hackl`_.


.. _`Jürgen Hackl`: hackl.j@gmx.at



Purpose
=======

PyRe (Python Reliability) is a python module for structural reliability
analysis. Its flexibility and extensibility make it applicable to a large
suite of problems. Along with core reliability analysis functionality, PyRe
includes methods for summarizing output.

Note
----

   At the moment First-Order Reliability Methods Crude Monte Carlo Simulation
   and Importance Sampling are supported! Second-Order Reliability Methods and
   some different sensitivity analysis will hopefully follow soon :)



History
=======

The FERUM (Finite Element Reliability Using Matlab) project was initiated in
1999 at the University of California, Berkeley, by Terje Haukaas and Armen Der
Kiureghian, primarily for pedagogical purposes aimed at teaching and learning
structural reliability and stochastic finite elements methods. [DerKiureghian2006]
This code consists of an open-source Matlab toolbox, featuring various
structural reliability methods. The latest available version (version 3.1),
which can be downloaded from `FERUM`_. Since 2003, this code is no longer
officially maintained. [Bourinet2010]

A new version of this open-source code (FERUM 4.x) based on a work carried out
at the Institut Français de Mécanique Avancée (`IFMA`_) in Clermont-Ferrand,
France. This version offers improved capabilities such as simulation-based
technique (Subset Simulation), Global Sensitivity Analysis (based on Sobol’s
indices), Reliability-Based Design Optimization (RBDO) algorithm, Global
Sensitivity Analysis and reliability assessment based on Support Vector
Machine (SVM) surrogates, etc. Beyond the new methods implemented in this
Matlab code. [Bourinet2009]

Of the purpose, to use structural reliability analysis for the project
"Generic Framework for Stochastic Modeling of Reinforced Concrete
Deterioration Caused by Corrosion" [Hackl2013] a python version of FERUM has
been created.


The focus here lies on the reliability analysis and not more on the finite
element method, so only the core function of FERUM are implemented.



Features
========

PyRe provides functionalities to make structural reliability analysis as easy
as possible. Here is a short list of some of its features:

* Perform reliability analysis with different kinds of Reliability Methods.

* Perform reliability analysis with Crude Monte Carlo Simulation.

* Includes a large suite of well-documented statistical distributions.

* Uses NumPy for numerics wherever possible.

* No limitation on the limit state function.

* Correlation between the random variables are possible.

* Traces can be saved to the disk as plain text.

* PyRe can be embedded in larger programs, and results can be analyzed
  with the full power of Python.


Getting started
===============

This `Documentation`_ provides all the information needed to install PyRe, code a
reliability model, run the sampler, save and visualize the results. In
addition, it contains a list of the statistical distributions currently
available.

.. _`Documentation`: http://hackl.github.io/pyre/

.. _`FERUM`: http://www.ce.berkeley.edu/projects/ferum/

.. _`IFMA`: http://www.ifma.fr/Recherche/Labos/FERUM


List of References
==================

[Bourinet2009] J.-M. Bourinet, C. Mattrand, and V Dubourg. A review of recent features and improvements added to FERUM software. In Proc. of the 10th International Conference on Structural Safety and Reliability (ICOSSAR’09), Osaka, Japan, 2009.

[Bourinet2010] J.-M. Bourinet. FERUM 4.1 User’s Guide, 2010.

[DerKiureghian2006] A. Der Kiureghian, T. Haukaas, and K. Fujimura. Structural reliability software at the University of California, Berkeley. Structural Safety, 28(1-2):44–67, 2006.

[Hackl2013] J. Hackl. Generic Framework for Stochastic Modeling of Reinforced Concrete Deterioration Caused by Corrosion. Master’s thesis, Norwegian University of Science and Technology, Trondheim, Norway, 2013.
