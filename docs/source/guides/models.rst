Specifying a reliability model
==============================

An analysis has three parts: a probability model for inputs, a limit-state
function for the event, and an algorithm with its options. Keep physical units
and the reference period in the model description: PySTRA does not infer them.

Name inputs explicitly
----------------------

Variable and constant names must match the limit-state function's arguments.
Most PySTRA marginal constructors take the physical mean and standard deviation,
or instead their native parameters as keywords, such as
``Gumbel("Q", loc=8.9, scale=1.56)``.
:class:`~pystra.distributions.scipy_dist.ScipyDist` wraps an already
parameterised SciPy distribution. Check the relevant distribution's signature.

.. testcode:: model

   import pystra as ra

   def limit_state(R, S, resistance_scale):
       return resistance_scale * R - S

   model = ra.StochasticModel()
   model.add_variable(ra.Normal("R", 10.0, 1.0))
   model.add_variable(ra.Normal("S", 5.0, 1.0))
   model.add_variable(ra.Constant("resistance_scale", 1.0))
   response = ra.LimitState(limit_state)
   result = ra.FORM(model, response).run()
   assert result.converged

Here ``R`` and ``S`` use the same force unit. A negative limit state means
failure. Write elementwise NumPy expressions so the function also accepts
batches of input values; avoid Python ``if`` statements on arrays. The
:doc:`/notebooks/ex_ddm` example explains the analytical-gradient return contract.
A limit state can also wrap an external solver and evaluate one point at a
time; see :doc:`/notebooks/ex_openseespy`.

Specify dependence before choosing a transformation
---------------------------------------------------

Independent inputs are the simplest starting point. A marginal distribution
for each input and a correlation matrix alone do not specify every possible
joint distribution. The traditional Gaussian/Nataf route interprets a
:class:`~pystra.dependence.correlation.CorrelationMatrix` as physical Pearson correlations;
its modified normal-space correlations are computed internally.

For a different dependence model, supply a complete
:class:`~pystra.dependence.joint.JointDistribution` with an explicit copula. Copula parameters
and physical Pearson correlations are different inputs. See :doc:`/copulas` for
supported transformations and conditioning order, and preserve variable order
when supplying matrices, samples and gradients.

Check the physical event
------------------------

Evaluate a safe and a failed case by hand. Record how load statistics relate
to annual, point-in-time or maximum-over-period events. For systems, write the
component events and their topology before selecting a solver. For load
combinations, retain the assumptions connecting leading and companion actions.

**Continue:** :doc:`/notebooks/ex_intro` · :doc:`/api/models` ·
:doc:`/api/probability` · :doc:`/theory/fundamentals`
