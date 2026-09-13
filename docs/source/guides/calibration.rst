Calibrating code factors
========================

Code calibration proposes factors, generates code-conforming designs, assesses
their reliability and revises the factors. PySTRA's normalized-reliability
workflow makes those steps explicit. Start with
:doc:`/notebooks/ex_generic_calibration` for a complete executable study.

Specify the model and candidate factors
---------------------------------------

:class:`~pystra.calibration.normalized.NormalizedReliabilityModel` holds the
resistance, load and model-error distributions, together with
:class:`~pystra.calibration.normalized.NominalValues`. Define characteristic or
nominal values consistently with the code being studied; they are not
interchangeable with means. State the load processes and reference period.

:class:`~pystra.calibration.normalized.CodeFactors` holds the resistance factor
``phi`` and action factors ``gamma_g``, ``gamma_p`` and ``gamma_q``. Name the
factor sets by the design rules they represent, for example current and
proposed rules.

Assess a design population
---------------------------

:class:`~pystra.calibration.normalized.CodeCalibration` evaluates the chosen
live-load and dead-load ratio grid. Call ``study.run(model, factors,
target_beta=...)`` for each candidate set, check ``result.converged``, and use
``result.to_frame()`` to inspect individual designs. The target is an input to
the study; selecting it requires the consequence and reference-period rationale.

Compare the full reliability surface and its low-reliability regions, not only
an average index. :func:`~pystra.calibration.plotting.plot_calibration` compares result
records and can show the applicable design ranges. The tutorial's annotated
ranges are illustrative assumptions, not prescribed domains for a bridge code.

Use another reliability method
-------------------------------

``CodeCalibration.run``, ``solve_designs`` and ``verify_designs`` accept
``evaluator=`` and evaluator-specific ``options=``. The default is FORM.
The evaluator receives an isolated model and limit state, and returns either
an analysis with ``run()`` or a reliability result. See the runnable analytic
adapter in :doc:`assessment` for the callback and result contract. The same
adapter convention is used by ``pystra.assessment.analyze_case``; the existing
``pystra.calibration.analyze_case`` import remains available.

Every grid point or load case is retained. Tables include ``method``,
``status``, ``converged`` and ``message``. Failed or under-resolved estimates
are NaN in summary tables, with the original record available on each case;
failed cases have no target margin. ``AnalysisError`` becomes a failed case,
while invalid inputs and programming errors still raise.

Derive and verify partial factors
---------------------------------

When factors are to be derived from selected design points, use the explicit
operations in :mod:`pystra.calibration.factors`. This is another step in code
calibration, with its own derivation and verification examples in
:doc:`/notebooks/ex_factor_calibration`.

For leading and companion actions, begin with
:doc:`/notebooks/ex_load_combinations`; each named case should contain the
actual distributions used in that analysis. Verify final factors on the
representative load combinations and design population, and report unsuccessful
analyses alongside the reliability range.

Root target solving and verification need a normal-equivalent reliability
estimate. Alpha projection additionally needs a finite, named physical design
point and direction in independent normal space, plus the analysis's matching
``transform.u_to_x`` and ``model``. Coefficient and matrix factor derivation
require physical design points in normal space. These operations reject an
estimate-only adapter clearly; they do not infer a FORM design point from a
probability estimate. Use ``solutions.reliability_frame()`` to inspect every
target solve, including inner reliability status, outer solve status and
residual. ``solutions.to_frame()`` remains the successful design-point table.

The load-case constructor takes ``cases=`` with explicit ``VariableRoles`` and
``leading_actions`` where factor methods need them. Case names come from that
mapping; there is no separate label list or legacy dictionary constructor.
Roles and the limit-state callable are read-only after validation. Construct
a new case specification to change them, or use isolated named overrides when
evaluating a design. Factor selection still requires a separate reliability
verification of the resulting designs; extrema alone do not certify a target.

**Continue:** :doc:`/api/calibration` · :doc:`/theory/code_calibration` ·
:doc:`/notebooks/ex_target_reliability`
