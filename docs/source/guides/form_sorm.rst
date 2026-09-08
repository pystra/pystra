Running FORM and SORM
=====================

FORM returns a convergence record and a local approximation to failure
probability. Start with the :doc:`/notebooks/ex_first_analysis` example, then use
:doc:`/notebooks/ex_intro` to compare approximations on a nonlinear model.

Run and inspect FORM
--------------------

.. testcode:: form-sorm

   import pystra as ra

   model = ra.StochasticModel()
   model.add_variable(ra.Normal("R", 10.0, 1.0))
   model.add_variable(ra.Normal("S", 5.0, 1.0))
   response = ra.LimitState(lambda R, S: R - S)
   form = ra.Form(stochastic_model=model, limit_state=response)
   result = form.run()
   assert result.converged, result.message
   assert result.failure_probability is not None
   design = dict(zip(result.variable_names, result.design_point))
   assert abs(design["R"] - design["S"]) < 1e-3

Inspect convergence before reading the probability or design point. Check that
the physical point lies on the intended boundary and that its units and signs
make sense. Convergence only establishes the local numerical solution: it does
not rule out other important failure regions. See :doc:`/strong_maximum` and
:doc:`troubleshooting` for further checks.

Reuse the analysis in SORM
--------------------------

Pass the completed ``Form`` analysis to SORM so it can reuse the design point
and transformation. The immutable ``FormResult`` is a reporting record; the
``form=`` argument requires the analysis object.

.. testcode:: form-sorm

   sorm = ra.Sorm(stochastic_model=model, limit_state=response, form=form)
   sorm.run(fit_type="cf")
   assert sorm.results_valid
   probability = float(sorm.pf2_breitung)
   assert abs(probability - result.failure_probability) < 1e-8

``fit_type="cf"`` uses curvature from the Hessian; ``"pf"`` fits boundary
points. Read ``pf2_breitung`` for Breitung's probability and
``pf2_breitung_m`` for the Hohenbichler–Rackwitz modification. Here the boundary
is a plane, so FORM and SORM agree with the exact probability. For a curved
boundary, inspect fitting diagnostics and compare with an independent estimate.

SORM requires standard-normal coordinates. A Student-t copula can still be
mapped to independent normals through Rosenblatt; explicit spherical Student-t
coordinates are a different choice. See :doc:`/copulas`.

**Continue:** :doc:`results` · :doc:`/api/reliability` ·
:doc:`/theory/design_point_methods`
