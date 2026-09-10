.. _chap_strong_maximum:

Strong Maximum Test
===================

For worked examples, see :doc:`notebooks/ex_strong_maximum`; for the
derivation, see :ref:`theory_strong_maximum`.

``StrongMaximumTest`` checks for failure points away from a candidate FORM
design point. It is an optional diagnostic: it does not change the FORM
probability, find replacement design points, or certify global optimality.

The method follows Dutfoy and Lebrun [DutfoyLebrun2006]_, as described in the
`OpenTURNS theory documentation
<https://openturns.github.io/openturns/latest/theory/reliability_sensitivity/strong_maximum_test.html>`_.
Pystra implements the independent standard-normal case using NumPy and SciPy;
OpenTURNS is not a runtime dependency.

After FORM
----------

.. code-block:: python

   import numpy as np
   import pystra as ra

   model = ra.StochasticModel()
   model.add_variable(ra.Normal("X", 0.0, 1.0))
   model.add_variable(ra.Normal("Y", 0.0, 1.0))
   form = ra.FORM(
       stochastic_model=model,
       limit_state=ra.LimitState(lambda X, Y: 3.0 - X),
   )
   form.run()

   check = ra.StrongMaximumTest(
       form, importance_level=0.15, accuracy_level=3.0,
       confidence_level=0.99, seed=2026,
   )
   print(check.point_number)  # cost is available before run()
   check.run()
   print(check.status)       # no_competing_region_detected for this plane
   print(check.confidence_level)
   print(check.get_points())  # far failure points, in U-space by default

A converged ``FORM`` supplies the model, transformation and design point.
The original limit-state function is evaluated without gradients. The test
uses an independent evaluator and does not overwrite the FORM evaluator's
last inputs. Keep the stochastic model and limit-state definition unchanged
between FORM and the check. Function evaluation accounting remains cumulative
on the shared model; ``check.evaluation_count`` counts this diagnostic only.

Alternatively, specify a point explicitly:

.. code-block:: python

   check = ra.StrongMaximumTest(
       stochastic_model=model,
       limit_state=ra.LimitState(lambda X, Y: 9.0 - X**2),
       design_point=[3.0, 0.0],
       point_number=500, seed=2026,
   )
   check.run()
   print(check.status)  # competing_region_detected: also fails for X < -3
   competing_u = check.get_points("far_failure")
   competing_x = check.get_points("far_failure", uspace=False)
   competing_g = check.get_values("far_failure")

The explicit point must be a finite vector in the model's independent
standard-normal coordinates. ``analysis_options`` selects its transformation
and, for Rosenblatt, conditioning order. For a non-Gaussian copula, use
:ref:`Rosenblatt <chap_copulas>`; spherical Student-t Nataf space is rejected.
The test verifies a strictly safe origin and a boundary
residual no greater than ``get_e1() * abs(g(origin))``. It rejects candidates
near the origin and nonconverged FORM inputs. It does not independently solve
the constrained design-point problem for an explicit candidate.

Geometry and sampling
---------------------

Let :math:`u^*` be the candidate and :math:`\beta=\|u^*\|`. The standard-normal
density ratio at radius :math:`r` is
:math:`\exp[-(r^2-\beta^2)/2]`. Setting it to the importance level
:math:`0<\varepsilon<1` gives

.. math::

   R_\varepsilon = \sqrt{\beta^2-2\log\varepsilon},\qquad
   \delta_\varepsilon = R_\varepsilon/\beta-1.

Sample at the larger radius
:math:`R=\beta(1+\tau\delta_\varepsilon)`, where ``accuracy_level`` is
:math:`\tau>1`. This parameter controls enlargement, not a numerical error
bound. The implementation computes the radius increment without subtracting
nearly equal square roots.

Independent Gaussian vectors are normalized onto this sphere. Each sampled
point is transformed back to physical variables and evaluated in batches.
A point is classified as near the candidate when
:math:`u^T(u^*/\beta)>\beta`. Failure uses Pystra's strict :math:`g<0`
convention. The four masks are ``near_failure``, ``far_failure``, ``near_safe``
and ``far_safe``. Points have shape ``(point_number, nrv)`` and values have
shape ``(point_number,)``. Both U and X coordinates are retained explicitly.

The reference detection cap has angular radius
:math:`\theta=\arccos(R_\varepsilon/R)`. This is different from the vicinity
angle :math:`\arccos(\beta/R)`. For dimension :math:`d>1`, its surface fraction
is the normalized spherical cap area

.. math::

   p_{\rm cap}=\frac12 I_{\sin^2\theta}\left(\frac{d-1}{2},\frac12\right).

For one variable the sphere consists of two points and
:math:`p_{\rm cap}=1/2`. Independent samples miss a fixed cap with probability
:math:`(1-p_{\rm cap})^N`. For requested nominal confidence :math:`c`, Pystra
therefore chooses

.. math::

   N=\left\lceil\frac{\log(1-c)}{\log(1-p_{\rm cap})}\right\rceil.

``confidence_level`` reports :math:`1-(1-p_{\rm cap})^N` for the actual count.
Specify either ``confidence_level`` or ``point_number``; omitting both selects
0.99. Pystra rounds the required count upward. The reviewed OpenTURNS
implementation rounds to nearest, which can fall below the requested nominal
confidence. For its two-dimensional reference with
:math:`\beta=\sqrt{10}-0.3`, :math:`\varepsilon=0.01`, :math:`\tau=2` and
:math:`c=0.999999`, Pystra uses 55 sphere points instead of 54. The radius and
vicinity geometry match the published reference.

Cost and interpretation
-----------------------

``max_points`` defaults to one million sphere evaluations. An oversized test
raises before allocating the sample or calling the model. It does not silently
reduce the confidence. Each successful run also evaluates the origin and
candidate, for ``point_number + 2`` sample-point evaluations in total. Samples
are retained, so storage is proportional to ``point_number * nrv``. A seeded
local NumPy generator makes fresh test instances reproducible without changing
the global random stream; rerunning an instance advances its generator.

For :math:`\beta=3`, :math:`\varepsilon=0.15`, :math:`\tau=3` and 0.99 nominal
confidence, the required counts are 111 in five dimensions, 1297 in ten and
125521 in twenty. The budget should be inspected before calling an expensive
structural model.

``competing_region_detected`` means at least one far failure point was found.
Those points are useful starting locations for further design-point searches
or for planning multiple-centre importance sampling. They are not themselves
optimized design points. Their raw limit-state magnitudes cannot quantify
importance: multiplying a limit state by a positive constant preserves the
event while changing those magnitudes.

``no_competing_region_detected`` means the sample found none. The nominal
confidence is a cap-detection probability under local-plane and failure-region
extent assumptions. It is neither a posterior probability that FORM is correct
nor a bound on its probability error. A bounded failure island entirely inside
the sampled sphere can be missed. Near safe points can also reveal departure
from the tangent-plane geometry. Inspect the point groups and use simulation
when the geometry is uncertain.

With system reliability
-----------------------

Run the check separately on the component ``FORM`` objects retained by
``SystemFORM``:

.. code-block:: python

   # system_form is an already-run ra.SystemFORM object
   checks = {}
   for name, component_form in system_form.component_results.items():
       test = ra.StrongMaximumTest(component_form, point_number=1000, seed=2026)
       test.run()
       checks[name] = test

This tests whether a component may have additional important regions beyond
its one tangent plane. It does not validate the whole system probability.
Multiple component modes are already intentional in system FORM; applying a
single-candidate check to the combined system event will naturally flag them.
Pystra does not run these additional evaluations automatically.

**Continue:** :doc:`notebooks/ex_strong_maximum` · :doc:`api/reliability` · :doc:`theory/design_point_methods`
