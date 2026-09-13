Code calibration and load combinations
======================================

Use ``NormalizedReliabilityModel``, ``CodeFactors`` and ``CodeCalibration`` for code calibration using normalized reliability. Factor derivation and verification are explicit operations in ``pystra.calibration.factors``.

Public entry points
-------------------

.. list-table::
   :header-rows: 1

   * - Object
     - Purpose
   * - :class:`~pystra.calibration.normalized.NormalizedReliabilityModel`
     - Specify normalized resistance, actions and model errors.
   * - :class:`~pystra.calibration.normalized.CodeFactors`
     - Name a candidate resistance/action factor set.
   * - :class:`~pystra.calibration.normalized.NominalValues`
     - Retain nominal values independently of distributions.
   * - :class:`~pystra.calibration.normalized.CodeCalibration`
     - Assess code-conforming designs over load ratios.
   * - :class:`~pystra.loads.LoadCombination`
     - Define explicit named reliability cases.
   * - :class:`~pystra.loads.FBCProcess`
     - Generate leading and companion action distributions.

**Use it:** :doc:`/guides/calibration` · :doc:`/notebooks/ex_generic_calibration` · :doc:`/theory/code_calibration`

Module details
--------------

.. autosummary::
   :toctree: ../gen
   :template: custom-module-template.rst
   :recursive:

   pystra.loads
   pystra.calibration
