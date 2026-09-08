Models and analysis options
===========================

Define random variables, limit states, model constants and algorithm options.

Public entry points
-------------------

.. list-table::
   :header-rows: 1

   * - Object
     - Purpose
   * - :class:`~pystra.model.StochasticModel`
     - Collect named variables, constants and dependence.
   * - :class:`~pystra.model.LimitState`
     - Wrap the physical response and gradient contract.
   * - :class:`~pystra.analysis.AnalysisOptions`
     - Set numerical tolerances, transformation and sampling options.

**Use it:** :doc:`/guides/models` · :doc:`/notebooks/ex_first_analysis` · :doc:`/theory/fundamentals`

Module details
--------------

.. autosummary::
   :toctree: ../gen
   :template: custom-module-template.rst
   :recursive:

   pystra.model
   pystra.analysis
