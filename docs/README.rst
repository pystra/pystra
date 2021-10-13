************
Introduction
************

Purpose
=======

Pystra (Python Structural Reliability Analysis) is a python module for structural reliability analysis. Its flexibility and extensibility make it applicable to a large
suite of problems. Along with core reliability analysis functionality, Pystra
includes methods for summarizing output. Pystra is also closely integrated with the usual python scientific packages workflow, numpy and scipy; in particular, all
statistical distributions in Scipy can be used in reliability modeling.

History
=======

The FERUM (Finite Element Reliability Using Matlab) project was initiated in
1999 at the University of California, Berkeley, by Terje Haukaas and Armen Der
Kiureghian, primarily for pedagogical purposes aimed at teaching and learning
structural reliability and stochastic finite elements methods. [DerKiureghian2006]_
This code consists of an open-source Matlab toolbox, featuring various
structural reliability methods. The latest available version (version 3.1),
which can be downloaded from `FERUM`_. Since 2003, this code is no longer
officially maintained. [Bourinet2010]_

A new version of this open-source code (FERUM 4.x) based on a work carried out
at the Institut Français de Mécanique Avancée (`IFMA`_) in Clermont-Ferrand,
France. This version offers improved capabilities such as simulation-based
technique (Subset Simulation), Global Sensitivity Analysis (based on Sobol’s
indices), Reliability-Based Design Optimization (RBDO) algorithm, Global
Sensitivity Analysis and reliability assessment based on Support Vector
Machine (SVM) surrogates, etc. Beyond the new methods implemented in this
Matlab code. [Bourinet2009]_


To use structural reliability analysis for the project
"Generic Framework for Stochastic Modeling of Reinforced Concrete
Deterioration Caused by Corrosion" [Hackl2013]_ a python version of FERUM was
created, called PyRe. PyRe was developed to v5.0.3 over 8 years and with its
success and wider adoption evolved into Pystra to make it more widely available
as a benchmark implementation of structural reliability analysis in python.


Features
========

Pystra provides functionalities to make structural reliability analysis as easy
as possible. Here is a short list of some of its features:

* Perform reliability analysis with different kinds of Reliability Methods.

* Includes a large suite of well-documented statistical distributions.

* Includes a wrapper for all statistical distributions supported in Scipy.

* Uses NumPy for numerics wherever possible.

* No limitation on the form of the limit state function

* Correlation between the random variables are possible

* Traces can be saved to the disk as plain text.

* Pystra can be embedded in larger programs, and results can be analyzed
  with the full power of Python.


Getting started
===============

This guide provides all the information needed to install Pystra, code a
reliability model, run the sampler, save and visualize the results. 

.. _`FERUM`: http://www.ce.berkeley.edu/projects/ferum/

.. _`IFMA`: http://www.ifma.fr/Recherche/Labos/FERUM
