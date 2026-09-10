Reliability algorithms and results
==================================

Run FORM, SORM, simulation, sensitivity and system-reliability analyses.

Public entry points
-------------------

.. list-table::
   :header-rows: 1

   * - Object
     - Purpose
   * - :class:`~pystra.form.FORM`
     - Find a design point and return convergence diagnostics.
   * - :class:`~pystra.sorm.SORM`
     - Fit local curvature around a converged FORM point.
   * - :class:`~pystra.mc.CrudeMonteCarlo`
     - Estimate the physical event by direct sampling.
   * - :class:`~pystra.mc.ImportanceSampling`
     - Concentrate weighted samples near a FORM point.
   * - :class:`~pystra.ls.LineSampling`
     - Estimate probability through line intersections.
   * - :class:`~pystra.ss.SubsetSimulation`
     - Reach rare events through conditional samples.
   * - :class:`~pystra.system_form.SystemFORM`
     - Combine component tangent models in a shared space.
   * - :class:`~pystra.results.FORMResult`
     - Retain an immutable FORM result.

**Use it:** :doc:`/guides/methods` · :doc:`/notebooks/ex_intro` · :doc:`/theory/design_point_methods`

Module details
--------------

.. autosummary::
   :toctree: ../gen
   :template: custom-module-template.rst
   :recursive:

   pystra.form
   pystra.results
   pystra.sorm
   pystra.mc
   pystra.ls
   pystra.ss
   pystra.sensitivity
   pystra.system
   pystra.system_form
   pystra.strong_maximum
