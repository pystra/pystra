Comparing structural assessment scenarios
=========================================

Represent each assessment scenario through explicit resistance, action and
model-error assumptions. Run the same physical failure event for each scenario,
then compare reliability and, where appropriate, engineering decisions.
PySTRA supplies the reliability and decision calculations; the engineering
basis for each scenario belongs in the assessment record.

Keep scenario assumptions visible
---------------------------------

Record geometry, units, evidence supporting the input distributions,
dependence, and the load reference period. For example, a measured section
loss may justify a different resistance model. A restricted loading policy may
justify a different action model. Rebuild the stochastic model for each case
so those changes are explicit and reproducible.

.. testcode:: assessment

   import pystra as ra
   from pystra.assessment import AssessmentCase, assess_cases

   def scenario(name, resistance_mean):
       model = ra.StochasticModel()
       model.add_variable(ra.Normal("R", resistance_mean, 1.0))
       model.add_variable(ra.Normal("S", 5.0, 1.0))
       return AssessmentCase(
           name, model, ra.LimitState(lambda R, S: R - S),
           metadata={
               "units": "kN", "reference_period_years": 50,
               "dependence": "independent", "model_error": "omitted for teaching",
           },
       )

   cases = [scenario("Reference", 10.0), scenario("Reduced resistance", 9.0)]
   result = assess_cases(cases)
   table = result.to_frame()
   assert result.converged
   assert table.loc[1, "beta"] < table.loc[0, "beta"]
   assert table.loc[0, "metadata"]["reference_period_years"] == 50

Each case owns a copied model and failure-event specification. The returned
record keeps every scenario, its assumptions and the method's diagnostics.
``result.to_frame()`` returns a fresh table; its ``converged``, ``status`` and
``message`` columns distinguish successful estimates from failed cases. Failed
estimates appear as NaN in the table, while the original reliability record
remains available in ``result.cases``. A failed case has no target margin.

These normal inputs are a teaching model with a common force unit. Choosing
probabilistic models from inspection or monitoring data is a separate modeling
step; this example does not implement Bayesian updating or deterioration over
time. See :doc:`models` and :doc:`results` for the analysis contract.

Choose a reliability evaluator
------------------------------

Assessment and calibration use FORM by default. Supply ``evaluator=`` to use
another suitable method or a callback for an external solver. The call is
``evaluator(model, limit_state, options=options)`` with copied inputs. A method
constructor such as ``ra.CrudeMonteCarlo`` returns an analysis whose ``run()``
is called once; a callback can return a
:class:`~pystra.assessment.ReliabilityEstimate` directly. The evaluator owns
its random state and external resources. For example, pass
``functools.partial(ra.CrudeMonteCarlo, rng=431)`` with ``SimulationOptions``
for a reproducible simulation.

This analytic adapter applies only to the independent normal R-minus-S
teaching model above:

.. testcode:: assessment

   import numpy as np
   from pystra.assessment import ReliabilityEstimate

   def normal_difference(model, limit_state, *, options=None):
       resistance, action = model.variable("R"), model.variable("S")
       beta = (resistance.mean - action.mean) / np.hypot(resistance.std, action.std)
       return ReliabilityEstimate(method="analytic normal difference", beta=beta)

   analytic = assess_cases(cases, evaluator=normal_difference)
   np.testing.assert_allclose(analytic.to_frame().beta, table.beta, atol=1e-6)

An adapter must return a normal-equivalent index and probability for the same
failure event and reference period. ``ReliabilityEstimate`` derives either
quantity when only the other is supplied; give both to retain a finite index
when a tail probability underflows. Sampling results that exhaust their budget
without meeting precision remain unsuccessful in a study, even if their raw
record includes an estimate. Review that estimate's uncertainty explicitly.

``AnalysisError`` becomes a failed case, preserving its record or message.
Invalid model specifications and programming errors still raise. A callback
that converts a failed solver run to a bare probability has discarded its
status: return the complete result or an unsuccessful ``ReliabilityEstimate``.

Connect reliability to a decision
---------------------------------

For a design or intervention grid,
:class:`~pystra.decision.ddo.DesignStudy` evaluates a reliability callback at candidate
values. ``study.run()`` returns a structured snapshot; ``study.evaluate()``
returns its table, including convergence, status, method and message.
:class:`~pystra.decision.ddo.DDO` combines the study with an acceptability
criterion and an optional objective. Distinguish the economic optimum from a feasible
optimum under that criterion, as demonstrated in
:doc:`/notebooks/ex_design_decision_optimization`. Failed alternatives remain
in the decision table and cannot be selected by ``optimize()`` or
``economic_optimum()``. ``DDO.run()`` clears the previous table before a rerun;
its returned table and ``DDO.results`` are copies. A later successful run is
required to replace an unsuccessful rerun.

State consequences, costs, discounting, jurisdictional inputs and reference
period before interpreting a target reliability. Published SWTP records are
dated inputs with provenance; check their suitability for the study rather
than treating them as current valuations. A target index is not transferable
between periods without assumptions about temporal dependence.

**Continue:** :doc:`/api/decisions` · :doc:`/theory/decisions` ·
:doc:`/notebooks/ex_target_reliability`
