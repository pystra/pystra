Interpreting results and convergence
====================================

Report the event, probability model, method, convergence status and numerical
precision together. A failure probability without its reference period or
modelling assumptions is incomplete.

Result records
--------------

Every analysis's ``run()`` returns an immutable record from
:mod:`pystra.results`. Its values remain a snapshot if the analysis is rerun,
and its vector fields are read-only arrays in ``variable_names`` order.

.. list-table:: Fields common to every record
   :header-rows: 1

   * - Field
     - Meaning
   * - ``status``, ``message``, ``converged``
     - How the method terminated. ``converged`` is true for the statuses
       ``"converged"`` and ``"completed"``.
   * - ``failure_probability``, ``beta``
     - The estimate and its normal-equivalent index, :math:`-\Phi^{-1}(p_f)`.
       Diagnostic records (Strong Maximum Test, distribution analysis) have
       neither.
   * - ``n_limit_state_evaluations``
     - Limit-state evaluations made by this run.
   * - ``method``, ``variable_names``
     - The analysis, and the order of the vector fields.

``summary()`` returns a short plain-text report.

.. list-table:: Status values
   :header-rows: 1

   * - ``status``
     - Meaning
   * - ``"converged"``
     - A design-point method (FORM, SORM, system FORM, sensitivity) met its
       convergence criteria.
   * - ``"not_converged"``
     - It did not, or SORM's formula is undefined at the fitted curvatures.
       There is no FORM or SORM estimate.
   * - ``"completed"``
     - A simulation or diagnostic ran to completion.
   * - ``"precision_not_met"``
     - A simulation used its sample budget before reaching its target
       coefficient of variation. The estimate is reported, with its
       coefficient of variation.

FORM records
------------

:class:`~pystra.results.FORMResult` adds:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``design_point_x``
     - The design point in physical coordinates and original units.
   * - ``design_point_u``, ``standard_space``
     - The same point in the chosen standard coordinates.
   * - ``alpha``
     - The unit direction from the origin towards the design point.
   * - ``design_index``
     - Signed distance of the design point; equals ``beta`` in normal space.
   * - ``iterations``, ``limit_state_error``, ``direction_error``
     - Diagnostics for the iteration and its termination.

``to_dataframe()`` tabulates the design point and ``alpha`` by variable. An
unconverged record has no probability, reliability index or design point.
The analysis object's ``get_beta()`` returns the design index, whereas
``get_equivalent_beta()`` and the record's ``beta`` are normal-equivalent.
This distinction matters for explicit spherical Student-t transformations;
see :doc:`/theory/notation`.

Other records
-------------

.. list-table::
   :header-rows: 1

   * - Record
     - Returned by, and what it adds
   * - :class:`~pystra.results.SORMResult`
     - SORM: ``form``, ``fit``, ``curvatures``, ``formula`` and
       ``approximations``, the Breitung and modified Breitung probabilities.
   * - :class:`~pystra.results.SimulationResult`
     - Crude Monte Carlo, importance, line and subset sampling:
       ``coefficient_of_variation``, ``n_samples`` and method-specific
       ``diagnostics``.
   * - :class:`~pystra.results.SystemFORMResult`
     - System FORM: ``bounds``, ``component_results``, ``correlation`` and
       ``intersections``.
   * - :class:`~pystra.results.SensitivityResult`
     - Sensitivity analysis: ``marginal`` derivatives and, for the closed
       form, ``correlation`` derivatives, with ``to_dataframe()``.
   * - :class:`~pystra.results.StrongMaximumResult`
     - Strong Maximum Test: a diagnostic, not a probability, with the sample
       points by region.
   * - :class:`~pystra.results.DistributionAnalysisResult`
     - Distribution analysis: the samples and their limit-state values.

Calibration and active-learning analyses have their own records. Retain their
convergence and termination information when exporting tables.

Separate three sources of error
-------------------------------

* **Numerical convergence:** did the iteration or fit meet its criteria?
* **Approximation error:** does a tangent plane, quadratic surface or surrogate
  represent the important failure regions?
* **Sampling uncertainty:** how variable is an estimate based on finite samples?

A confidence interval on a frozen surrogate's Monte Carlo estimate addresses
the last item, conditional on that surrogate. It does not measure its bias.
A completed calculation or a small coefficient of variation does not establish
the physical model's adequacy.

**Continue:** :doc:`troubleshooting` · :doc:`/plotting` ·
:doc:`/notebooks/ex_first_analysis` · :doc:`/api/reliability`
