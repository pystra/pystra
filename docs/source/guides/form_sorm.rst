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
   form = ra.FORM(model, response)
   result = form.run()
   assert result.converged, result.message
   assert result.failure_probability is not None
   design = dict(zip(result.variable_names, result.design_point_x))
   assert abs(design["R"] - design["S"]) < 1e-3

Inspect convergence before reading the probability or design point. Check that
the physical point lies on the intended boundary and that its units and signs
make sense. Convergence only establishes the local numerical solution: it does
not rule out other important failure regions. See :doc:`/strong_maximum` and
:doc:`troubleshooting` for further checks.

Reuse the analysis in SORM
--------------------------

Pass the completed ``FORM`` analysis to SORM so it can reuse the design point
and transformation. The immutable ``FORMResult`` is a reporting record; the
``form=`` argument requires the analysis object.

.. testcode:: form-sorm

   sorm_result = ra.SORM(model, response, form=form).run()
   assert sorm_result.converged
   probability = sorm_result.failure_probability
   assert abs(probability - result.failure_probability) < 1e-8

``SORMOptions(fit="curve")``, the default, uses curvature from the Hessian;
``fit="point"`` fits boundary points. ``sorm_result.approximations`` gives
Breitung's probability (``"breitung"``, the estimate by default) and the
Hohenbichler–Rackwitz modification (``"modified_breitung"``). Here the boundary
is a plane, so FORM and SORM agree with the exact probability. For a curved
boundary, inspect fitting diagnostics and compare with an independent estimate.

SORM requires standard normal coordinates. A Student-t copula can still be
mapped to independent normals through Rosenblatt; explicit spherical Student-t
coordinates are a different choice. See :doc:`/copulas`.

**Continue:** :doc:`results` · :doc:`/api/reliability` ·
:doc:`/theory/design_point_methods`
