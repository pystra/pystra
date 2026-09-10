Troubleshooting reliability analyses
====================================

FORM does not converge
----------------------

Read ``FORMResult.message``, ``iterations``, ``limit_state_error`` and
``direction_error``. Check the limit-state sign, units and values near the trial
point. Nonfinite values, a zero gradient or a discontinuity require attention
to the response model or method choice. Verify analytical derivatives against
finite differences. Review starting points and tolerances before increasing the
iteration budget. See :doc:`form_sorm` and :doc:`/notebooks/ex_ddm`.

FORM converges but methods disagree
-----------------------------------

Convergence describes the design-point iteration. Curvature, competing failure
regions and transformation ordering can change the approximation error. Inspect
the physical design point, compare an independent simulation, and use
:doc:`/strong_maximum` where applicable. The test can miss bounded failure
islands, so an empty set of competing points is not a global guarantee.

Monte Carlo observes zero failures
----------------------------------

Do not interpret a zero count as proof that failure is impossible. For a
preselected, fixed sample size of ``N`` independent direct trials with no
failures, a one-sided 95% binomial upper confidence limit is
:math:`1-0.05^{1/N}` (approximately :math:`3/N`). This formula does not apply
unchanged to importance weights, correlated subset chains or adaptive stopping.
Increase the budget or choose a rare-event estimator using :doc:`methods`.

Sampling estimates change between runs
--------------------------------------

Record seeds, budgets and achieved precision, then repeat independent runs.
For subset simulation, inspect intermediate thresholds and chain behaviour;
the reported ``cov`` neglects chain correlation. For surrogate methods, separate
training variability, surrogate bias and final sampling variability. See
:doc:`simulation` and :doc:`/active_learning`.

A transformation is unsupported
-------------------------------

Check the copula, variable order and chosen standard space together. A
Student-t copula does not require Student-t standard coordinates. Rosenblatt
maps supported continuous joint laws to independent standard normals, which
are required by several classical algorithms. Explicit spherical Student-t
coordinates have different algorithm support. See :doc:`/copulas`.

Active learning stops at its budget
-----------------------------------

Inspect the stopping reason and convergence record; reaching the evaluation
limit is not evidence that the surrogate meets the convergence criterion.
Reserve independent evaluations for validation. Review the learning function,
surrogate fit and estimator coverage in :doc:`/active_learning`.

A notebook cannot import an object
----------------------------------

Confirm the notebook kernel uses the environment where the development version
is installed. Inspect ``sys.executable``, ``pystra.__version__`` and
``pystra.__file__`` inside that kernel. Install the ``al`` extra for the
active-learning tutorials and retain any helper files in the runnable bundle.
See :doc:`/install` and :doc:`/migrating` for environment and API changes.
