Load combinations and FBC processes
***********************************

.. _id1:


Load combination reliability problems usually distinguish permanent actions,
resistance variables, and variable actions that fluctuate in time.  The
Ferry-Borges-Castanheta (FBC) model represents a variable action as a
rectangular-wave stochastic process: the process is constant during a basic
interval :math:`\tau`, and a new independent value is drawn for each
successive interval.  This model is a standard basis for probabilistic load
combination analysis in structural reliability texts [Thoft-Christensen]_
[Madsen2006]_ [Ditlevsen2007]_ [Melchers1999]_.

If :math:`F_Q(q)` is the distribution of the action value in one basic
interval and :math:`T` is the reference period, the maximum over that period
has distribution

.. math::
   :label: eq:fbc_max_distribution

   F_{Q,\max,T}(q) = F_Q(q)^r,
   \qquad r = \frac{T}{\tau}

where :math:`r` is the number of basic intervals in the reference period.
Equivalently, when a code or statistical model supplies a maximum
distribution over duration :math:`T`, the corresponding maximum over a
shorter duration :math:`d` can be written as

.. math::
   :label: eq:fbc_companion_distribution

   F_{Q,\max,d}(q) = F_{Q,\max,T}(q)^{d/T}

provided both maxima arise from the same FBC process assumptions.  The
recurrence count belongs to the underlying stochastic process, not to the
load-combination factor itself.

Turkstra's rule is a practical approximation for combining variable actions:
each variable action is taken as the leading action in turn, usually as a
maximum over the reference period, while the other variable actions are taken
as companion values over a representative interval.  In an FBC setting a
companion action can be the point-in-time value or the maximum over the
leading action's basic interval.  The modelling distinction is important:
the FBC process defines the distribution of each action over time; Turkstra's
rule defines which distributions are placed together in each reliability
case.  This is the convention followed in Sørensen's notes and common load
combination examples [Sorensen2004]_ [Faber2009]_.

In Pystra, :class:`~pystra.fbc.FBCProcess` exposes the process distributions:
``point_in_time()`` returns the basic-interval parent distribution, and
``maximum(duration=...)`` returns a maximum distribution for the requested
duration.  :meth:`~pystra.loadcomb.LoadCombination.turkstra` then uses those
process objects to create explicit named leading-action cases.  The result is
still an ordinary :class:`~pystra.loadcomb.LoadCombination`; the generated
cases simply make the FBC and Turkstra assumptions visible in the model.

**Use this method:** :doc:`/guides/calibration` · :doc:`/notebooks/ex_load_combinations` · :doc:`/api/calibration`

For coordinate conventions, see :doc:`notation`.
