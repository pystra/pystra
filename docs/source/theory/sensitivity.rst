Reliability sensitivity
***********************

Sensitivity analysis
====================


In structural reliability, knowing the reliability index :math:`\beta` alone
is often insufficient. Engineers also need to understand *how sensitive*
:math:`\beta` is to the parameters of the stochastic model — the means,
standard deviations, and correlation coefficients of the random variables.
This information guides decisions about where to invest in data collection
or quality control.

PySTRA computes the sensitivity
:math:`\partial\beta/\partial\theta_k` for each distribution parameter
:math:`\theta_k` using two complementary approaches.

Finite-Difference method
------------------------

The simplest approach perturbs each parameter by a small amount
:math:`\Delta\theta_k = \delta\,\sigma_k` and re-runs FORM:

.. math::
   :label: eq:fd_sens

   \frac{\partial\beta}{\partial\theta_k}
   \approx \frac{\beta(\theta_k + \Delta\theta_k) - \beta(\theta_k)}
               {\Delta\theta_k}

With :math:`m` declared distribution parameters this requires :math:`m + 1`
FORM runs: one baseline and one perturbed run per parameter, or :math:`2n + 1`
when each of :math:`n` variables declares only its mean and standard deviation.
The method is straightforward and distribution-agnostic, but
can be numerically unstable when the perturbation changes the Nataf
transformation significantly — particularly for correlated non-normal
variables with small sensitivities.

Closed-Form method (Bourinet 2017)
----------------------------------

A more efficient and accurate approach post-processes the converged FORM
design point to obtain exact (up to quadrature) sensitivities from a
single FORM run. This method, due to [Bourinet2017]_ (building on the
FERUM software framework [Bourinet2009]_ [Bourinet2010]_), differentiates
the Nataf transformation chain analytically.

The sensitivity of :math:`\beta` to a marginal distribution parameter
:math:`\theta_k` decomposes into two terms:

.. math::
   :label: eq:cf_sens

   \frac{\partial\beta}{\partial\theta_k}
   = \underbrace{{\boldsymbol\alpha}^T \mathbf{L}_0^{-1}
     \frac{\partial\mathbf{z}}{\partial\theta_k}}_{\text{first term}}
   + \underbrace{{\boldsymbol\alpha}^T
     \frac{\partial\mathbf{L}_0^{-1}}{\partial\theta_k}
     \mathbf{z}}_{\text{second term}}

where :math:`\boldsymbol\alpha` is the FORM direction cosine vector,
:math:`\mathbf{L}_0` is the Cholesky factor of the modified (Nataf)
correlation matrix :math:`\mathbf{R}_0`, and :math:`\mathbf{z}` is the
correlated standard-normal design point.

The first term captures how the marginal transformation changes at the
design point; the second term accounts for changes in the correlation
structure due to the parameter perturbation. For uncorrelated normal
variables, the second term vanishes identically.

The derivative of the inverse Cholesky factor is computed from:

.. math::
   :label: eq:dinvL

   \frac{\partial\mathbf{L}_0^{-1}}{\partial\theta}
   = -\mathbf{L}_0^{-1}\,
     \frac{\partial\mathbf{L}_0}{\partial\theta}\,
     \mathbf{L}_0^{-1}

where :math:`\partial\mathbf{L}_0/\partial\theta` is obtained by
simultaneously differentiating the Cholesky decomposition algorithm.

Correlation sensitivities :math:`\partial\beta/\partial\rho_{ij}` are
also available from the closed-form method. Since the marginal
transformations do not depend on the correlation coefficients, only the
second term of Equation :eq:`eq:cf_sens` contributes.

Generalized parameter Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond mean and standard deviation, distributions may declare additional
sensitivity parameters — for example, the shape parameter :math:`\xi` of
the GEV distribution controls the tail behavior and can significantly
influence :math:`\beta`.

Each distribution declares its sensitivity parameters via the
:attr:`~pystra.distributions.distribution.Distribution.sensitivity_params`
property.  The base class returns ``{"mean", "std"}``; subclasses with
extra parameters (e.g. GEV shape) override this to include them.  The
sensitivity pipeline then iterates over whatever parameters each
distribution declares, so both the finite-difference and closed-form
methods generalize automatically.

For shape parameters, the partial derivatives
:math:`\partial F_X / \partial\theta` and
:math:`\partial\mu / \partial\theta`,
:math:`\partial\sigma / \partial\theta` are evaluated numerically via
central differences unless the distribution provides an analytical
override.  See the :ref:`developer guide <adding_distributions>` for
implementation details.

**Use this method:** :doc:`/guides/sensitivity` · :doc:`/notebooks/ex_sensitivity` · :doc:`/api/reliability`

For coordinate conventions, see :doc:`notation`.
