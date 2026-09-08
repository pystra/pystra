Reliability figures
===================

Create and customise figures from fitted models and completed results. See the :doc:`/plotting` guide for examples.

Public entry points
-------------------

.. list-table::
   :header-rows: 1

   * - Object
     - Purpose
   * - :func:`~pystra.plotting.plot_limit_state`
     - Compare physical and surrogate boundaries in two dimensions.
   * - :func:`~pystra.plotting.plot_form_geometry`
     - Inspect transformed boundaries and a design point.
   * - :func:`~pystra.plotting.plot_learning_history`
     - Inspect sequential enrichment and diagnostics.
   * - :func:`~pystra.plotting.plot_pce_selection`
     - Compare polynomial truncations and fit scores.
   * - :func:`~pystra.calibration.plotting.plot_calibration`
     - Compare reliability over load ratios.

**Use it:** :doc:`/plotting` · :doc:`/notebooks/ex_generic_calibration` · :doc:`/theory/code_calibration`

Module details
--------------

.. autosummary::
   :toctree: ../gen
   :template: custom-module-template.rst
   :recursive:

   pystra.plotting
