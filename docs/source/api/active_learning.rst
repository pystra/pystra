Active-learning reliability
===========================

Compose surrogates, learning functions, reliability estimators and stopping criteria. See the :doc:`/active_learning` guide for method selection and assumptions.

Public entry points
-------------------

.. list-table::
   :header-rows: 1

   * - Object
     - Purpose
   * - :class:`~pystra.active_learning.analysis.ActiveLearning`
     - Compose and run a surrogate-assisted analysis.
   * - :class:`~pystra.active_learning.surrogates.KrigingSurrogate`
     - Fit a Gaussian-process response model.
   * - :class:`~pystra.active_learning.surrogates.PceSurrogate`
     - Fit dense or adaptive sparse Hermite polynomials.
   * - :class:`~pystra.active_learning.pc_kriging.PcKrigingSurrogate`
     - Fit a polynomial trend with a Kriging residual.
   * - :class:`~pystra.active_learning.estimation.MonteCarloEstimator`
     - Estimate probability on a separate sample.
   * - :class:`~pystra.active_learning.importance.ImportanceSamplingEstimator`
     - Use explicit Gaussian-mixture proposal centres.
   * - :class:`~pystra.active_learning.subset.SubsetSimulationEstimator`
     - Use replicated conditional sampling.
   * - :class:`~pystra.active_learning.stopping.BetaBounds`
     - Stop on a surrogate beta-band criterion.

**Use it:** :doc:`/active_learning` · :doc:`/notebooks/ex_active_learning` · :doc:`/theory/active_learning`

Module details
--------------

.. autosummary::
   :toctree: ../gen
   :template: custom-module-template.rst
   :recursive:

   pystra.active_learning
