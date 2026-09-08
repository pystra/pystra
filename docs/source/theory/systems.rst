System reliability
******************

.. contents:: On this page
   :local:
   :depth: 2

System Reliability
==================

System reliability concerns a structure whose failure is governed by more than
one component event.  If the component limit states are
:math:`g_i({\bf X})`, the component failure events are

.. math::

   F_i = \{g_i({\bf X}) \leq 0\}.

For a series system the system failure event is the union of component failure
events,

.. math::

   F_\mathrm{series} = \bigcup_{i=1}^{n} F_i,

and the equivalent scalar limit-state function can be written as

.. math::

   g_\mathrm{series}({\bf X}) = \min_i g_i({\bf X}).

For a parallel system the system failure event is the intersection of component
failure events,

.. math::

   F_\mathrm{parallel} = \bigcap_{i=1}^{n} F_i,

with equivalent scalar limit-state function

.. math::

   g_\mathrm{parallel}({\bf X}) = \max_i g_i({\bf X}).

These min/max forms are useful because they preserve the standard Pystra sign
convention: positive means safe and non-positive means failed.  They also
allow the same system definition to be passed to simulation methods, active
learning, and, when the envelope is sufficiently smooth near the controlling
point, FORM/SORM.

k-of-n, Cut-Set, and Tie-Set Systems
------------------------------------

More general topologies are often described in terms of events rather than a
single analytic limit-state expression [Ditlevsen2007]_.  A k-of-n system
fails when at least :math:`k` component events have occurred:

.. math::

   F_{k|n} =
   \left\{\sum_{i=1}^{n} I(F_i) \geq k\right\},

where :math:`I(F_i)` is one if event :math:`F_i` occurs and zero otherwise.
This representation is exact for Boolean enumeration and simulation, but it is
not generally differentiable.

If the minimum cut sets :math:`C_m` are known, the system failure event can be
written as

.. math::

   F_\mathrm{sys} =
   \bigcup_{m=1}^{n_c} \left(\bigcap_{i \in C_m} F_i\right).

The dual path, or tie-set, representation writes the safe event as the union
of working tie sets.  For tie sets :math:`T_m`,

.. math::

   S_\mathrm{sys} =
   \bigcup_{m=1}^{n_t} \left(\bigcap_{i \in T_m} \bar{F}_i\right).

Cut-set and tie-set descriptions are common in structural system reliability
because they let the engineer encode known collapse mechanisms or load paths
without enumerating every possible Boolean state [Song2003]_.

Ditlevsen Bounds
----------------

For a series system, exact evaluation of
:math:`P(\cup_i F_i)` may require high-dimensional integration over a union of
failure domains.  If the component probabilities :math:`P(F_i)` and pairwise
intersections :math:`P(F_i \cap F_j)` are available, Ditlevsen's bounds give a
second-order estimate of the union probability [Ditlevsen1979]_.  For a chosen
event ordering, the lower bound is

.. math::

   P_L =
   P(F_1) +
   \sum_{i=2}^{n}
   \max\left[
      P(F_i) - \sum_{j=1}^{i-1} P(F_i \cap F_j),\ 0
   \right],

and the upper bound is

.. math::

   P_U =
   \sum_{i=1}^{n} P(F_i)
   - \sum_{i=2}^{n} \max_{1 \leq j < i} P(F_i \cap F_j).

The bounds depend on event ordering.  For small systems the ordering can be
checked exhaustively; for large systems, the ordering should be chosen using
engineering judgement or a heuristic.  Mainçon's 100-element series-system
benchmark is a useful validation case because it reports component and
pairwise probabilities directly [Maincon2000]_.

Linear-programming bounds generalise this idea to arbitrary systems and
arbitrary available event information.  Song and Der Kiureghian showed that LP
bounds can use component, pairwise, and higher-order event probabilities for
general cut-set systems, including the rigid-plastic cantilever-bar benchmark
[Song2003]_.  This is a natural extension beyond the current Ditlevsen bounds
API.

Four-Branch Case
----------------

The four-branch case is a widely used benchmark for reliability algorithms
because it has multiple disconnected failure regions [Schueremans2005]_.  With
independent standard normal variables :math:`X_1` and :math:`X_2`, it is
defined by

.. math::

   g_\mathrm{FBC}({\bf X}) = \min(g_1, g_2, g_3, g_4),

where

.. math::

   \begin{aligned}
   g_1 &= 3 + 0.1(X_1 - X_2)^2 - \frac{X_1 + X_2}{\sqrt{2}}, \\
   g_2 &= 3 + 0.1(X_1 - X_2)^2 + \frac{X_1 + X_2}{\sqrt{2}}, \\
   g_3 &= (X_1 - X_2) + \frac{6}{\sqrt{2}}, \\
   g_4 &= (X_2 - X_1) + \frac{6}{\sqrt{2}}.
   \end{aligned}

The benchmark is a series system in event terms, but a single FORM analysis
can find only one local design point.  Simulation, subset simulation, and
active-learning methods are therefore better suited to estimating the global
failure probability unless a dedicated first-order system reliability method
is used.

First-Order System Reliability
------------------------------

First-order system reliability methods approximate each component failure
surface near its design point and then integrate the resulting system event in
standard normal space.  This requires more information than a scalar topology:
component design points, component normal vectors, dependence between
linearised events, and a clear isoprobabilistic transformation.  Rosenblatt
transformations add an additional ordering issue because the transformed
standard-space geometry can depend on the conditioning order [Meinen2025]_.

For this reason Pystra currently separates three tasks:

1. users encode the system topology using series, parallel, k-of-n, cut-set,
   or tie-set systems;
2. existing simulation and active-learning methods estimate the resulting
   failure probability directly;
3. analytical bounds such as Ditlevsen bounds are computed from event
   probabilities when those probabilities are available.
