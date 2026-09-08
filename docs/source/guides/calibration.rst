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

**Continue:** :doc:`/api/calibration` · :doc:`/theory/code_calibration` ·
:doc:`/notebooks/ex_target_reliability`
