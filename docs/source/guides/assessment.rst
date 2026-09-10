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

   def assess(resistance_mean):
       model = ra.StochasticModel()
       model.add_variable(ra.Normal("R", resistance_mean, 1.0))
       model.add_variable(ra.Normal("S", 5.0, 1.0))
       return ra.FORM(
           stochastic_model=model,
           limit_state=ra.LimitState(lambda R, S: R - S),
       ).run()

   scenarios = {"Reference": assess(10.0), "Reduced resistance": assess(9.0)}
   assert all(result.converged for result in scenarios.values())
   assert scenarios["Reduced resistance"].beta < scenarios["Reference"].beta

These normal inputs are a teaching model with a common force unit. Choosing
probabilistic models from inspection or monitoring data is a separate modelling
step; this example does not implement Bayesian updating or deterioration over
time. See :doc:`models` and :doc:`results` for the analysis contract.

Connect reliability to a decision
---------------------------------

For a design or intervention grid,
:class:`~pystra.ddo.DesignStudy` evaluates a reliability callback at candidate
values. :class:`~pystra.ddo.DDO` combines the study with an objective and optional
acceptability criterion. Distinguish the economic optimum from a feasible
optimum under that criterion, as demonstrated in
:doc:`/notebooks/ex_design_decision_optimization`.

State consequences, costs, discounting, jurisdictional inputs and reference
period before interpreting a target reliability. Published SWTP records are
dated inputs with provenance; check their suitability for the study rather
than treating them as current valuations. A target index is not transferable
between periods without assumptions about temporal dependence.

**Continue:** :doc:`/api/decisions` · :doc:`/theory/decisions` ·
:doc:`/notebooks/ex_target_reliability`
