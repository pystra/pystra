Literature benchmark definitions and provenance
===============================================

The :doc:`notebooks/ex_literature_truss` and
:doc:`notebooks/ex_literature_hat` tutorials complement the existing
:doc:`notebooks/ex_active_learning` four-branch example. They compare
classical approximations and current active-learning components using
independent probability references. These are reproducible problem definitions
and PySTRA algorithm variants, not replications of every published setting.

Use the runnable-bundle download on either notebook page to include its helper,
or download :download:`literature_benchmarks.py <notebooks/literature_benchmarks.py>`
separately. The helper is
example code, not a new public modelling API. Only NumPy, SciPy and PySTRA
are needed for these two tutorials; their bootstrap-PCE examples do not
require the optional Kriging dependency.

23-bar truss
------------

Marelli and Sudret (2018), Section 3.2, Figure 4 and Tables 2–3 define the
geometry, independent distributions and 0.12 m downward midspan displacement
limit. There are six 4 m panels, a depth of 2 m, a left pin and a right roller.
Eleven horizontal members share one stiffness product; twelve diagonals share
another. The six loads act downwards at the upper nodes. The tutorial uses
the exact linear-elastic unit-load expression in place of repeated FE solves.
Tests independently assemble the planar truss stiffness matrix from geometry
and check random inputs and all six individual load cases.

The paper's million-sample Monte Carlo estimate is 1.52e-3. Its binomial
standard error is approximately 3.9e-5; this estimate is not an exact target.
Our separate conditional integration removes one Gumbel load analytically
and uses independent scrambled Sobol replications for the remaining variables.
It supplies an integration reference without fitting a surrogate or using
PySTRA probability transformations. Replication error is reported separately.

The tutorial reports the paper's FORM, SORM and A-bPCE values alongside
PySTRA's results. FORM and curve-fitting modified Breitung (Hohenbichler–Rackwitz) SORM agree
at the paper's precision. The unmodified Breitung formula gives a different
approximation, also shown. This numerical agreement does not establish which
SORM formula the authors used; the manuscript does not specify it.
The active-PCE variant uses a Latin hypercube initial design, single-point
U enrichment, normal-coordinate Hermite polynomials and combined beta stopping
criteria. The paper uses a uniform-ball initial design, three-point enrichment,
a bootstrap classification criterion and different sampling settings. Evaluation
counts therefore describe this tutorial run, not a performance ranking against
the paper's algorithms.

Hat function
------------

The explicit definition comes from the UQLab 2.2.0 hat reliability example:
independent normal inputs with mean 0.25 and standard deviation 1, and
``g = 20 - (x1 - x2)**2 - 8*(x1 + x2 - 4)**3``.
Rotation into independent sum and difference coordinates reduces its failure
probability to one-dimensional quadrature, giving 3.865398086e-4. The tutorial
explains the integral and checks a cubic surrogate away from its training data.
Exact representability makes this a useful polynomial and transformation check;
it is not evidence that PCE will be exact for general structural models.

Table 3 of the Moustapha–Marelli–Sudret review's supplementary material
(arXiv:2106.01713v2, page 46) lists 4.40e-3 for “Hat function” and 3.85e-4
for “4-branch series”. These do not match the explicitly defined UQLab hat
and standard k=6 four-branch problems (the latter has probability 4.45733e-3).
The values appear consistent with a transposition, but we have not established
that this is the explanation. We retain the printed values as source metadata
and use independently derived references for the explicit definitions. We do
not tune a model to force agreement or claim to correct the published table.

Source code basis
-----------------

The ignored local UQLab installation remains unchanged. The following files
informed the analytical expressions and geometry checks. Copyright (c)
2018–2026 Stefano Marelli and Bruno Sudret (ETH Zurich); the full BSD-3-Clause
notice is retained in ``THIRD_PARTY_NOTICES``. Adaptations use vectorized Python,
a positive downward displacement convention, separate model factories and
new independent integration/FE checks. No UQLab runtime is required.

.. list-table:: UQLab 2.2.0 source files (relative to its root)
   :header-rows: 1
   :widths: 60 40

   * - Source
     - SHA-256

   * - ``Examples/SimpleTestFunctions/uq_TrussModelAnalytical.m``
     - ``e56f6b3d7fb967abcd43e5f566c33c3f8224b61d70d5cd6537f1709d24e2dd00``

   * - ``Examples/SimpleTestFunctions/Truss_Matlab_FEM/uq_truss_model.m``
     - ``2bf6fba3062a0eb57bfe31574fbb44b61536b294d3e4344b2c3548c5047b092e``

   * - ``Examples/SimpleTestFunctions/uq_hat.m``
     - ``eacd4bed87ce783482cdb033f52548cdfe2d5d6cdcf715c3742b388c5be7b16a``

   * - ``Examples/Reliability/uq_Example_Reliability_02_hat.m``
     - ``b13f992dd198619c0f9989275d25a20e389809f54165fb126ed7e97dc087dd22``

References
----------

* Marelli, S. and Sudret, B. (2018). An active-learning algorithm that combines
  sparse polynomial chaos expansions and bootstrap for structural reliability
  analysis. Structural Safety 75, 67–74.
  https://doi.org/10.1016/j.strusafe.2018.06.003
  (`open manuscript <https://arxiv.org/abs/1709.01589>`__).
* Moustapha, M., Marelli, S. and Sudret, B. (2022). Active learning for structural
  reliability: survey, general framework and benchmark. Structural Safety 96,
  102174. https://doi.org/10.1016/j.strusafe.2021.102174
  (`manuscript and supplement <https://arxiv.org/abs/2106.01713>`__).
* Teixeira, R., Nogal, M. and O’Connor, A. (2021). Adaptive approaches in
  metamodel-based reliability analysis: A review. Structural Safety 89, 102019.
  https://doi.org/10.1016/j.strusafe.2020.102019. This complementary review
  motivates coverage across method families; it is not the numerical source
  of the two problems here.

**Continue:** :doc:`benchmarks` · :doc:`tutorials/active_learning` · :doc:`citing`
