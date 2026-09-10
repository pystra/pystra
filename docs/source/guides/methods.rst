Choosing a reliability method
=============================

Start with the failure event, its joint probability model and the cost of
one limit-state evaluation. A converged algorithm can still give an inaccurate
approximation. Use an independent reference or a second method where feasible.

.. list-table:: Starting points and checks
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Useful starting point
     - What to check
   * - :class:`~pystra.form.FORM`
     - Smooth response with a dominant failure region; economical initial study.
     - Convergence, gradients, competing design points and local approximation error.
   * - :class:`~pystra.sorm.SORM`
     - Curvature near a converged FORM design point matters.
     - Curvature and fitting validity; additional failure regions remain a concern.
   * - :class:`~pystra.mc.CrudeMonteCarlo`
     - Affordable evaluations; a direct check on the physical failure event.
     - Enough observed failures and reported sampling precision.
   * - :class:`~pystra.mc.ImportanceSampling`
     - A FORM design point identifies an important failure region.
     - Proposal coverage, weight variability and regions away from that point.
   * - :class:`~pystra.ls.LineSampling`
     - A useful direction is available and line intersections can be found.
     - Direction choice, root searches and boundary geometry along lines.
   * - :class:`~pystra.ss.SubsetSimulation`
     - Rare events for which direct Monte Carlo is too costly.
     - Threshold progression, chain mixing and variability across independent runs.
   * - :class:`~pystra.system_form.SystemFORM`
     - A system can be described through component limit states and topology.
     - Every component's convergence and a shared transformation; validate the original event.
   * - :class:`~pystra.active_learning.analysis.ActiveLearning`
     - Expensive evaluations justify fitting and enriching a surrogate.
     - Training budget, stopping reason, independent validation and final estimator uncertainty.

A practical sequence
--------------------

1. Verify units, the sign of failure and the joint distribution using
   :doc:`models` and :doc:`/copulas`.
2. For a smooth component, run :doc:`form_sorm` and inspect the design point.
   For a system, preserve the topology and use :doc:`/system`.
3. Check the approximation with :doc:`simulation` or an independent integral
   on a tractable benchmark. A small FORM/SORM difference alone is insufficient.
4. When evaluations dominate cost, consider :doc:`/active_learning`, retaining
   an evaluation budget for independent validation.

Discontinuous functions can be sampled directly when affordable, but ordinary
FORM and SORM require local derivatives. Strong nonlinearity, multiple failure
regions and transformation ordering require particular care. PySTRA normally
uses independent standard-normal coordinates; see :doc:`/theory/notation`.

**Continue:** :doc:`/benchmarks` · :doc:`results` · :doc:`/api/reliability`
