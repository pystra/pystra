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
   * - :class:`~pystra.distributions.scipydist.ScipyDist`
     - Wrap a parameterised SciPy distribution.
   * - :class:`~pystra.joint.JointDistribution`
     - Combine marginals with an explicit copula.
   * - :class:`~pystra.copula.GaussianCopula`
     - Specify Gaussian dependence.
   * - :class:`~pystra.copula.StudentTCopula`
     - Specify Student-t dependence and degrees of freedom.
   * - :class:`~pystra.copula.FrankCopula`
     - Specify bivariate Frank dependence.

**Use it:** :doc:`/copulas` · :doc:`/notebooks/ex_copulas` · :doc:`/theory/transformations`

Module details
--------------

.. autosummary::
   :toctree: ../gen
   :template: custom-module-template.rst
   :recursive:

   pystra.distributions
   pystra.copula
   pystra.joint
   pystra.transformation
   pystra.correlation
   pystra.integration
   pystra.quadrature
