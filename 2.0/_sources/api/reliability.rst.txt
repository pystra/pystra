Reliability algorithms and results
==================================

Run FORM, SORM, simulation, sensitivity and system-reliability analyses.

Public entry points
-------------------

.. list-table::
   :header-rows: 1

   * - Object
     - Purpose
   * - :class:`~pystra.reliability.form.FORM`
     - Find a design point and return convergence diagnostics.
   * - :class:`~pystra.reliability.sorm.SORM`
     - Fit local curvature around a converged FORM point.
   * - :class:`~pystra.reliability.monte_carlo.CrudeMonteCarlo`
     - Estimate the physical event by direct sampling.
   * - :class:`~pystra.reliability.importance_sampling.ImportanceSampling`
     - Concentrate weighted samples near a FORM point.
   * - :class:`~pystra.reliability.line_sampling.LineSampling`
     - Estimate probability through line intersections.
   * - :class:`~pystra.reliability.subset_simulation.SubsetSimulation`
     - Reach rare events through conditional samples.
   * - :class:`~pystra.reliability.system_form.SystemFORM`
     - Combine component tangent models in a shared space.
   * - :mod:`pystra.results`
     - Immutable records returned by every analysis's ``run()``.

**Use it:** :doc:`/guides/methods` · :doc:`/notebooks/ex_intro` · :doc:`/theory/design_point_methods`

Module details
--------------

.. autosummary::
   :toctree: ../gen
   :template: custom-module-template.rst
   :recursive:

   pystra.reliability.form
   pystra.results
   pystra.reliability.sorm
   pystra.reliability.monte_carlo
   pystra.reliability.importance_sampling
   pystra.reliability.line_sampling
   pystra.reliability.subset_simulation
   pystra.reliability.sensitivity
   pystra.systems
   pystra.reliability.system_form
   pystra.reliability.strong_maximum
