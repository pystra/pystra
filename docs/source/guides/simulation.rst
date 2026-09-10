Estimating failure probability by simulation
============================================

Use simulation to estimate the probability of the specified failure event and
to check local approximations. The cost depends on the failure probability,
limit-state evaluation cost and precision required. Begin with the comparison
in :doc:`/notebooks/ex_simulation`.

Budget a direct Monte Carlo run
-------------------------------

``SimulationOptions(n_samples=...)`` sets the maximum sample budget for crude
Monte Carlo; ``target_cov`` can end the run earlier when its precision
criterion is met. Pass ``rng`` a seed for a reproducible run; NumPy's global
random state is neither used nor changed. Record the seed and actual
evaluation count in a study.

.. testcode:: simulation

   import numpy as np
   import pystra as ra

   model = ra.StochasticModel()
   model.add_variable(ra.Normal("R", 3.0, 1.0))
   model.add_variable(ra.Normal("S", 0.0, 1.0))
   options = ra.SimulationOptions(n_samples=20_000, target_cov=0.05)
   analysis = ra.CrudeMonteCarlo(
       model, ra.LimitState(lambda R, S: R - S), options=options, rng=2026
   )
   result = analysis.run()
   assert result.status in ("completed", "precision_not_met")
   assert 0 < result.failure_probability < 1

Check precision, not just probability
-------------------------------------

For a fixed number of independent direct samples, the approximate relative
standard error is :math:`\sqrt{(1-p_f)/(N p_f)}`. Rare failures therefore need
many evaluations. A zero failure count is insufficient evidence for a zero
failure probability; see :doc:`troubleshooting`.

Importance sampling concentrates samples using a proposal distribution and
weights their contributions. The proposal must cover the important failure
regions. The traditional :class:`~pystra.reliability.importance_sampling.ImportanceSampling` uses a FORM-based
centre; the separate active-learning estimator supports explicit proposal
components, as shown in :doc:`/notebooks/ex_active_extensions`.

Line sampling uses an important direction and searches for intersections along
lines. Subset simulation reaches a rare event through intermediate events and
conditional Markov chains. Inspect the thresholds and repeat independent runs
to assess stability: the current subset-simulation ``cov`` ignores chain
correlation and is a lower-bound precision diagnostic, not a calibrated
confidence interval.

**Continue:** :doc:`results` · :doc:`/api/reliability` ·
:doc:`/theory/simulation`
