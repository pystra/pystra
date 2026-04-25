.. _chap_system:

******************
System Reliability
******************

Pystra supports simple structural system composition through the
``pystra.system`` module.  The module does not introduce a separate
probability approximation method.  It builds an equivalent scalar
limit-state function from named component limit states, and that function can
then be used with the existing FORM, SORM, Monte Carlo, line sampling, subset
simulation, and active learning workflows.

The sign convention is the usual Pystra convention: positive values are safe
and negative values indicate failure.

Series and Parallel Systems
===========================

A series system fails when any child component or subsystem fails.  Its
equivalent limit-state value is therefore

.. math::

   g_\mathrm{series}(x) = \min_i g_i(x).

A parallel system fails only when all child components or subsystems fail.  Its
equivalent limit-state value is therefore

.. math::

   g_\mathrm{parallel}(x) = \max_i g_i(x).

Systems can be nested, so a user can describe a known structural topology
without enumerating every failure path manually.  This includes common branch
system benchmarks such as the four-branch series example [Schueremans2005]_.

Example
=======

.. code-block:: python

   import pystra as ra

   system = ra.SeriesSystem(
       [
           ra.Component("flexure", lambda R, S: R - S),
           ra.Component("shear", lambda V, S: V - 2.0 * S),
           ra.ParallelSystem(
               [
                   ra.Component("backup_a", lambda A: A),
                   ra.Component("backup_b", lambda B: B),
               ]
           ),
       ],
       name="frame",
   )

   limit_state = system.as_limit_state()

   model = ra.StochasticModel()
   model.addVariable(ra.Normal("R", 10.0, 1.0))
   model.addVariable(ra.Normal("V", 12.0, 1.0))
   model.addVariable(ra.Normal("S", 4.0, 1.0))
   model.addVariable(ra.Normal("A", 1.0, 0.5))
   model.addVariable(ra.Normal("B", 1.0, 0.5))

   form = ra.Form(stochastic_model=model, limit_state=limit_state)
   form.run()

Component functions may use only the stochastic variables they need.  Extra
model variables are filtered at the component boundary, which keeps small
component functions reusable inside larger systems.

Event Topologies
================

Structural systems are often specified as event logic once the engineer has
identified the relevant component limit states.  Pystra provides three small
helpers for that layer:

``KofNSystem``
   fails when at least ``k`` children fail.  ``KofNSystem(children, k=1)`` has
   the same failure event as a series system, and ``KofNSystem(children, k=n)``
   has the same failure event as a parallel system.

``CutSetSystem``
   fails when any supplied cut set has fully failed.  This is useful when the
   user already knows the minimal cut sets of the structure.

``TieSetSystem``
   remains safe when any supplied tie set remains fully safe.  This is useful
   for path-based descriptions of redundant systems.

For example, the rigid-plastic cantilever-bar benchmark used by
[Song2003]_ has the system failure event

.. math::

   E_\mathrm{sys} = E_1 E_2 \cup E_3 E_4 \cup E_3 E_5.

This can be represented directly from named components:

.. code-block:: python

   components = {
       "E1": ra.Component("E1", lambda T, X: T - 5.0 * X / 16.0),
       "E2": ra.Component("E2", lambda M, L, X: M - L * X),
       "E3": ra.Component("E3", lambda M, L, X: M - 3.0 * L * X / 8.0),
       "E4": ra.Component("E4", lambda M, L, X: M - L * X / 3.0),
       "E5": ra.Component("E5", lambda M, L, T, X: M + 2.0 * L * T - L * X),
   }

   system = ra.CutSetSystem(
       [["E1", "E2"], ["E3", "E4"], ["E3", "E5"]],
       components=components,
   )

Event-counting and cut/tie-set systems preserve the correct failure sign for
simulation, subset simulation, active learning, and Boolean enumeration.  They
are not generally smooth limit-state functions, so FORM/SORM results should be
interpreted with care unless the topology reduces to a smooth min/max envelope.

Ditlevsen Bounds
================

When component event probabilities and pairwise intersection probabilities are
available, :func:`pystra.ditlevsen_bounds` computes Ditlevsen's second-order
bounds for a union of failure events [Ditlevsen1979]_.  The event ordering can
be supplied explicitly, or exhaustively optimized for small systems.

.. code-block:: python

   probabilities = [0.1, 0.2]
   intersections = {(0, 1): 0.03}

   lower, upper = ra.ditlevsen_bounds(probabilities, intersections)

For two events the bounds collapse to the exact inclusion-exclusion result.
For larger systems they provide a cheap check on simulation estimates and a
useful validation target for future system FORM approximations.  The initial
test suite includes the identical 100-element series benchmark from
[Maincon2000]_, where the Ditlevsen upper bound is approximately
``6.216e-2``.

Scope and Transformations
=========================

The system module composes limit-state functions in the original physical
variables.  The isoprobabilistic transformation to standard space remains the
responsibility of the selected Pystra analysis method and its
``AnalysisOptions``.  This keeps system topology separate from the probability
transformation, following the same conceptual split used in structural
reliability methods generally.

This first implementation deliberately avoids first-order system reliability
method approximations and automatic failure-path enumeration.  Those methods
can be useful, but they need additional assumptions about the component
design points, dependence model, and transformation ordering.  In particular,
Rosenblatt-based first-order system approximations can be sensitive to the
conditioning order used for the transformation [Meinen2025]_.  For now,
simulation methods and active learning can estimate the probability of failure
from the composed system limit-state function directly.

Validation Benchmarks
=====================

The current validation suite checks exact Boolean behaviour for series,
parallel, k-of-n, cut-set, and tie-set systems; integration with Pystra's
``LimitState`` evaluation; a FORM smoke test for smooth min/max systems; and
Ditlevsen bounds for simple and published series-system cases.

The next benchmarks to add before extending the approximation methods are:

- the Ditlevsen frame examples in [Ditlevsen1979]_;
- the rigid-plastic cantilever-bar cut-set example and linear-programming
  bounds in [Song2003]_;
- the Daniels equal-load-sharing bundle examples in [Daniels1945]_;
- the correlated series-system cases in [Maincon2000]_ and related equivalent
  planes comparisons.
