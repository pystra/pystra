Distributions and dependence
============================

Specify marginals and dependence, and construct probability transformations.

Public entry points
-------------------

.. list-table::
   :header-rows: 1

   * - Object
     - Purpose
   * - :class:`~pystra.distributions.normal.Normal`
     - Specify a normal marginal by mean and standard deviation.
   * - :class:`~pystra.distributions.scipy_dist.ScipyDist`
     - Wrap a parameterised SciPy distribution.
   * - :class:`~pystra.dependence.joint.JointDistribution`
     - Combine marginals with an explicit copula.
   * - :class:`~pystra.dependence.copula.GaussianCopula`
     - Specify Gaussian dependence.
   * - :class:`~pystra.dependence.copula.StudentTCopula`
     - Specify Student-t dependence and degrees of freedom.
   * - :class:`~pystra.dependence.copula.FrankCopula`
     - Specify bivariate Frank dependence.

**Use it:** :doc:`/copulas` · :doc:`/notebooks/ex_copulas` · :doc:`/theory/transformations`

Module details
--------------

.. autosummary::
   :toctree: ../gen
   :template: custom-module-template.rst
   :recursive:

   pystra.distributions
   pystra.dependence.copula
   pystra.dependence.joint
   pystra.dependence.transformation
   pystra.dependence.correlation
