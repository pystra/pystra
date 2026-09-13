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
     - Wrap a parameterized SciPy distribution.
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

Directed transformation derivatives
-----------------------------------

Both ``Transformation`` and ``CopulaTransformation`` provide
``jacobian_u_wrt_x(u, x, marg)`` and ``jacobian_x_wrt_u(u, x, marg)``.
The joint transformation already owns its marginals, so ``marg`` is optional
there. Inputs are corresponding reference and physical points. Both matrices
have shape ``(dimension, dimension)`` with output coordinates in rows and input
coordinates in columns, in original model variable order. They are inverses
at nonsingular points. ``standard_space`` identifies the reference law: normal
Rosenblatt coordinates are independent; generalized Nataf with a Student-t
copula has spherical, dependent Student-t coordinates.

Distribution reconstruction
---------------------------

``dist.parameters`` is a read-only constructor mapping;
``type(dist)(**dist.parameters)`` reconstructs the marginal.
``dist.with_parameters(**changes)`` returns an independent updated marginal,
including its nested distributions. Native reconstruction parameters and
``sensitivity_params`` are separate: see :doc:`/development/distributions`
for moment updates, fixed bounds and custom subclass requirements.
