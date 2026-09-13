Working at high reliability
===========================

Structural target reliability indices depend on the reference period and the
consequence class, and optimization, assessment and rare-event studies can reach
much larger indices. The probabilities involved are then tiny, and ordinary
floating-point formulas fail before the reliability method does. The standard
normal CDF rounds to exactly one above :math:`u \approx 8.3`, so the usual
transformation :math:`x = F^{-1}(\Phi(u))` returns an infinite load there.
PySTRA evaluates each tail probability on the side where it is small, and
switches to log probabilities where even that underflows. This page shows what
you can rely on and how to read results far into the tails.

What stays accurate
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Behavior at small probabilities
   * - Marginal transformations
     - In an unbounded tail, accurate to floating-point tolerance while the
       tail probability is a normal double, down to about
       :math:`2 \times 10^{-308}` at :math:`|u| \approx 37.5`: the built-in
       distributions round-trip there to within about :math:`10^{-10}` in
       :math:`u`. Beyond that for the normal, lognormal and Gumbel families,
       and for ``Maximum`` and ``MaxParent`` built on them. A finite bound
       limits resolution on its own side; see `Limits`_.
   * - Distribution functions
     - ``cdf``, ``sf``, ``logcdf``, ``logsf`` and ``logpdf`` of the built-in
       distributions are accurate in their own tails. ``ppf`` and ``isf`` of
       some SciPy distributions are numerical inverses that can fail far into
       a tail; the transformations check them (see `Custom distributions`_).
   * - FORM
     - Design points far into a tail transform correctly, and Jacobians use
       log densities where the densities underflow.
   * - SORM
     - Curvature corrections and the Mills ratio are evaluated in log space.
       The reliability index stays finite when the probability underflows;
       this is checked at :math:`\beta = 40` for both fits and formulas.
   * - Importance sampling
     - Likelihood ratios and weight sums are accumulated in log space. The
       reliability index and the estimated coefficient of variation of the
       probability estimate stay finite when the probability underflows; this
       is checked at :math:`\beta = 40`.
   * - Line sampling
     - Line contributions are pooled in log space. Each line is searched only
       while every standard normal coordinate, and for Nataf each correlated
       coordinate, stays within :math:`[-37, 37]`. A line whose crossing lies
       beyond raises :class:`~pystra.errors.AnalysisError`. A limit state
       :math:`40 - X` therefore fails, whereas
       :math:`40 - (X + Y)/\sqrt{2}`, with the same index, succeeds.

Numerical stability is separate from accuracy. FORM and SORM remain local
approximations, and sampling estimates remain uncertain; see :doc:`form_sorm`
and :doc:`simulation`.

Check a high-reliability case
-----------------------------

A lognormal resistance against a Gumbel load has its design point at
:math:`u \approx 8.9` for the load, where :math:`\Phi(u)` has already rounded
to one. FORM and line sampling agree with a numerical reference integral of
the model's one-dimensional formula:

.. testcode:: high-reliability

   import numpy as np
   from scipy import integrate
   from scipy.stats import norm
   import pystra as ra

   resistance = ra.Lognormal("R", 60.0, 2.0)
   load = ra.Gumbel("S", 8.0, 1.5)
   model = ra.StochasticModel()
   model.add_variable(resistance)
   model.add_variable(load)
   limit_state = ra.LimitState(lambda R, S: R - S)

   form = ra.FORM(model, limit_state)
   form_result = form.run()
   line = ra.LineSampling(
       model,
       limit_state,
       form=form,
       options=ra.SimulationOptions(n_samples=50),
       rng=1,
   ).run()

   # P(S > R): the resistance density times the load's survival function
   pf, _ = integrate.quad(
       lambda r: np.exp(resistance.logpdf(r) + load.logsf(r)),
       resistance.ppf(1e-12),
       resistance.isf(1e-15),
       points=[60.0],
       epsabs=0,
       epsrel=1e-10,
       limit=500,
   )
   print(
       f"FORM {form_result.beta:.3f}, line sampling {line.beta:.3f}, "
       f"reference {-norm.ppf(pf):.3f}"
   )
   print(f"load design point u = {form_result.design_point_u[1]:.2f}")

.. testoutput:: high-reliability

   FORM 9.002, line sampling 9.003, reference 9.003
   load design point u = 8.86

The failure probability here is about :math:`10^{-19}`. The load's
transformation stays finite and invertible even further out, where a direct
:math:`F^{-1}(\Phi(10))` returns infinity:

.. testcode:: high-reliability

   x = load.u_to_x(10.0)
   print(f"x = {x:.2f}, back to u = {load.x_to_u(x):.6f}")

.. testoutput:: high-reliability

   x = 69.58, back to u = 10.000000

Read a probability that underflows
----------------------------------

Doubles below about :math:`2 \times 10^{-308}` lose precision, and below about
:math:`5 \times 10^{-324}` they are zero. A probability that small, or one that
underflows in an intermediate step, is reported as ``0.0``. SORM, importance
sampling and line sampling compute the reliability index from the logarithm of
the probability, so the index remains finite. It is then the quantity to report:

.. testcode:: high-reliability

   standard = ra.StochasticModel()
   standard.add_variable(ra.Normal("X", 0, 1))
   standard.add_variable(ra.Normal("Y", 0, 1))
   result = ra.ImportanceSampling(
       standard,
       ra.LimitState(lambda X, Y: 40 - X),
       options=ra.SimulationOptions(n_samples=4000, target_cov=0),
       rng=1,
   ).run()
   print(result.failure_probability, round(result.beta, 2))

.. testoutput:: high-reliability

   0.0 40.0

Probabilities this small rarely matter for a structure directly. But
intermediate designs in optimization and sensitivity studies can reach them, and
a finite index keeps those studies well defined. Crude Monte Carlo with no
observed failures also reports a zero probability, with an infinite index and
uncertainty. That zero is a lack of information, not an underflow; see
:doc:`troubleshooting`.

Custom distributions
--------------------

A distribution built on a SciPy frozen distribution, such as
:class:`~pystra.distributions.scipy_dist.ScipyDist` or a subclass that passes
``dist_obj``, inherits SciPy's survival and log functions. A subclass that
overrides one of these functions should also override its companions:

* ``cdf`` or ``ppf``: also ``sf`` and ``isf``. Otherwise the upper tail falls
  back to ``1 - cdf(x)`` and ``ppf(1 - q)``, which lose it.
* ``pdf``: also ``logpdf``. Otherwise the log density is ``log(pdf(x))``, and
  the Jacobian fails where the density underflows.
* ``logcdf`` and ``logsf``, computed without first forming the probability,
  extend the transformation beyond :math:`10^{-308}`.

When the base class's ``u_to_x`` is used, a quantile with a tail probability
below :math:`10^{-8}` is checked against the log-CDF or log-survival function
and solved again if it misses by more than :math:`10^{-8}` in relative log
probability. This guards against numerical inverses that stall far into a tail,
as SciPy's beta quantile does. Direct calls to ``ppf`` or ``isf`` are not
checked, and distributions with their own transformations, such as ``Normal``
and ``Lognormal``, do not need the check.

Limits
------

* A finite bound cannot be approached closer than one unit in the last place.
  Bounded tails, as in ``Uniform``, ``Beta``, the lower bounds of
  ``ShiftedExponential`` and ``ShiftedLognormal``, and the bounded GEV forms,
  therefore lose resolution before the probability does: transformed points
  reach the bound itself.
* Without a closed form, the tails depend on SciPy's log functions. Some are
  computed as ``log(cdf)`` or ``log(sf)`` and underflow beyond
  :math:`|u| \approx 38`, as for ``Gamma``; others, such as the logistic's,
  stay accurate further, and the transformation then extends with them.
* Line sampling's search range is limited as described in the table.
* The conditional functions of non-elliptical copulas, such as Frank, are
  evaluated in probability space.
* ``ZeroInflated`` keeps points within its probability mass at exactly zero.

**Continue:** :doc:`simulation` · :doc:`troubleshooting` ·
:ref:`Theory: very small probabilities <very-small-probabilities>` ·
:doc:`/api/probability`
