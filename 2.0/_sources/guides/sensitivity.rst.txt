Computing reliability sensitivities
===================================

Use :class:`~pystra.reliability.sensitivity.SensitivityAnalysis` to estimate
how the FORM reliability index changes when a marginal distribution parameter
changes. The derivative :math:`\partial\beta/\partial\theta` describes a local
change around the specified model and FORM design point. It can help compare
small, practical changes in a mean, standard deviation or tail-shape parameter.

Choose the method
-----------------

Set ``method`` on construction, then call ``run()`` without arguments.
Both methods use :class:`~pystra.options.FORMOptions` for the underlying
design-point searches.

.. list-table:: Sensitivity approaches
   :header-rows: 1
   :widths: 22 38 40

   * - Method
     - Computation
     - Conditions and outputs
   * - ``"numerical"``, the default
     - One baseline FORM run plus one per declared marginal parameter
     - Forward differences; marginal derivatives only. Check perturbation size
       and convergence of every run.
   * - ``"closed_form"``
     - Post-process one FORM run using the differentiated Nataf transformation
     - Use physical Pearson correlation input with the default Cholesky
       transformation. Returns marginal and correlation derivatives.

The closed-form route requires the physical-Pearson model interface
(``model.set_correlation(...)``, or its independent default). Explicit
copulas and ``FORMOptions(transform="nataf")`` or ``"rosenblatt"`` are
rejected by that route. Use ``method="numerical"`` for marginal sensitivity
with an explicit copula supported by FORM; its copula parameters stay fixed.
See :doc:`/copulas` for the distinction between physical correlation and
copula parameters.

Check a resistance–load model
-----------------------------

Let :math:`R` and :math:`S` be independent normal resistance and load in
the same force unit, with
:math:`(\mu_R,\sigma_R)=(10,2)` and
:math:`(\mu_S,\sigma_S)=(5,1)`. Failure is :math:`g=R-S\leq0`.
The boundary is linear in standard normal space, so FORM is exact here:

.. math::

   \beta=\frac{\mu_R-\mu_S}{\sqrt{\sigma_R^2+\sigma_S^2}}
        =\sqrt{5}.

.. testcode:: sensitivity-guide

   import numpy as np
   import pystra as ra

   model = ra.StochasticModel()
   model.add_variable(ra.Normal("R", mean=10.0, std=2.0))
   model.add_variable(ra.Normal("S", mean=5.0, std=1.0))
   response = ra.LimitState(lambda R, S: R - S)
   result = ra.SensitivityAnalysis(model, response, method="closed_form").run()
   assert result.converged, result.message
   assert result.variable_names == ("R", "S")
   np.testing.assert_allclose(result.beta, np.sqrt(5.0), rtol=1e-10)

   expected = {
       "R": {"mean": 1 / np.sqrt(5.0), "std": -2 / np.sqrt(5.0)},
       "S": {"mean": -1 / np.sqrt(5.0), "std": -1 / np.sqrt(5.0)},
   }
   for name, derivatives in expected.items():
       for parameter, value in derivatives.items():
           np.testing.assert_allclose(result.marginal[name][parameter], value, rtol=1e-8)

``result.marginal["R"]["mean"]`` is about :math:`0.4472` per force unit:
increasing mean resistance by :math:`0.1` increases :math:`\beta` by about
:math:`0.0447` locally. Both standard-deviation derivatives are negative
in this example. To compare parameters with different units or plausible
changes, compare
:math:`(\partial\beta/\partial\theta)\,\Delta\theta` using explicit
increments :math:`\Delta\theta`.

The symmetric ``result.correlation`` matrix follows
``result.variable_names`` and has a zero diagonal. Each off-diagonal entry
is the derivative with respect to one physical Pearson correlation
coefficient; the mirrored entries describe the same coefficient. For this
example, differentiating
:math:`\beta(\rho)=5/\sqrt{5-4\rho}` at :math:`\rho=0` gives
:math:`2/\sqrt{5}`. Numerical quadrature introduces a small approximation
in the computed derivative:

.. testcode:: sensitivity-guide

   np.testing.assert_allclose(
       result.correlation,
       [[0.0, 2 / np.sqrt(5.0)], [2 / np.sqrt(5.0), 0.0]],
       rtol=1e-6,
       atol=1e-10,
   )

Compare finite differences
--------------------------

The numerical method perturbs a parameter by
``h = delta * distribution.std`` and reruns FORM. This rule applies to
every declared parameter, including dimensionless shape parameters, so
check the resulting increment and parameter bounds. A smaller ``delta``
reduces truncation error until solver and rounding errors become significant.

.. testcode:: sensitivity-guide

   for delta in (1e-3, 1e-4):
       numerical = ra.SensitivityAnalysis(
           model, response, method="numerical", delta=delta
       ).run()
       assert numerical.converged, numerical.message
       assert numerical.correlation is None
       for name, derivatives in expected.items():
           for parameter, value in derivatives.items():
               np.testing.assert_allclose(
                   numerical.marginal[name][parameter], value, rtol=1e-3
               )

Both step sizes agree with the analytic derivatives to within :math:`0.1\%`
in this example. For a nonlinear model, inspect stability across step sizes
and, where needed, tighten the FORM convergence settings. Each numerical run
here performs five FORM analyses: one baseline and four parameter perturbations.

Interpret the parameter contract and diagnostics
------------------------------------------------

A marginal's
:attr:`~pystra.distributions.distribution.Distribution.sensitivity_params`
declares the parameters being differentiated. Most built-ins expose ``mean``
and ``std``; ``GEV`` and ``GEVMin`` also expose ``shape``. A shape
derivative holds the other declared coordinates, including mean and standard
deviation, fixed. Reconstruction through ``parameters`` and
``with_parameters`` is a separate capability: an adapter or composite can
be copied without declaring sensitivity parameters. An empty marginal entry
therefore means no parameters were declared, not that all sensitivities are zero.
See :doc:`/development/distributions` for extension requirements.

``result.form`` records the baseline FORM solution, and
``result.n_limit_state_evaluations`` includes all FORM runs. The default
``on_failure="raise"`` raises ``AnalysisError`` if a run does not converge.
With ``on_failure="return"``, inspect ``result.converged`` and
``result.message`` before using the derivatives; an unsuccessful result has
no usable marginal sensitivities.

Sensitivity describes the local FORM approximation. It does not establish
that the design point dominates the failure event, nor measure global
variance-based importance. For the FORM probability
:math:`p_f=\Phi(-\beta)`, the corresponding local derivative is
:math:`\partial p_f/\partial\theta
=-\phi(\beta)\,\partial\beta/\partial\theta`.

**Continue:** :doc:`form_sorm` · :doc:`/notebooks/ex_sensitivity` ·
:doc:`/api/reliability` · :doc:`/theory/sensitivity`
