Active-learning methods and literature
======================================

Two complementary reviews guide PySTRA's coverage:

* [TeixeiraNogalOConnor2021]_ surveys adaptive response surfaces, polynomial
  chaos, support vector machines and Kriging, including the choices of
  experimental design, enrichment and stopping. Its journal year is 2021;
  the article was available online in 2020.
* [Moustapha2022]_ organizes active-learning reliability into four components:
  **surrogate model, reliability estimator, learning function and stopping
  criterion**. Its benchmark compares 39 strategies on 20 problems. The
  findings are conditional on those methods/problems, rather than a universal
  ranking. The journal year is 2022; its preprint appeared in 2021.

The second review's Section 5 and Table 2 provide a useful reason to prioritize
surrogate-assisted subset simulation and PC-Kriging. They also distinguish
learning-point selection from stopping based on the probability/reliability
estimate. Simply implementing a learning function does not reproduce the
complete algorithm that originally introduced it.

Surrogate-assisted reliability and the coherent component design are part of
the **v2.0 release scope**. Kriging's scikit-learn dependency remains an
optional installation extra. Development proceeds on the v2.0 branch.

Current coverage on the 2.0 branch
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Component
     - Implemented
     - Gaps
   * - Surrogates
     - Kriging; adaptive sparse Hermite PCE with selected-support bootstrap;
       explicit dense OLS option
     - PC-Kriging; adaptive response-surface workflows; SVM classification
   * - Reliability estimation
     - Independent final MC; replicated subset simulation with adaptive
       enrichment and independent final sampling
     - Active-learning integration with importance sampling
   * - Learning functions
     - Explicit selection interface; U and EFF
     - Bootstrap voting/fraction of bootstrap replicates (FBR); batch selection
   * - Stopping
     - Explicit policies for learning thresholds, beta bands, beta stability
       and combined criteria; separate final sampling precision;
       explicit budget/exhaustion status
     - Diagnostics for importance-weighted sampling

Standalone FORM, SORM, crude Monte Carlo, importance sampling, line sampling
and subset simulation already exist. Having these solvers does not mean that
active learning can compose every standalone solver. ``ActiveLearning`` now
composes a replicated subset estimator through ``EnrichmentEstimator``.
The existing standalone subset solver remains a comparison baseline, with its
original documented CoV limitation. Active importance sampling remains future
work.

The fixed-pool Kriging/U loop is an AK-MCS-style implementation. EFF is provided
as a learning function; this is not a complete EGRA reproduction. Likewise,
the PCE implementation reproduces specified UQLab sparse-selection routines
and fast-bootstrap behavior, with documented numerical differences, but uses
PySTRA's single-point enrichment and stopping contract. Dense quadratic PCE
can span quadratic response surfaces in normal coordinates, but does not
supply a classical adaptive
response-surface reliability algorithm.

Composing the implemented methods
---------------------------------

The public import path remains ``pystra.active_learning``. Internally,
surrogates, learning functions, estimation, stopping and results have separate
modules. For example, with a PySTRA ``model`` and ``limit_state``::

   from pystra.active_learning import (
       ActiveLearning, KrigingSurrogate, MonteCarloEstimator, UFunction,
       AllCriteria, LearningThreshold, BetaBounds, BetaStability,
   )

   analysis = ActiveLearning(
       stochastic_model=model,
       limit_state=limit_state,
       surrogate=KrigingSurrogate(seed=7),
       estimator=MonteCarloEstimator(n_samples=100_000),
       learning_function=UFunction(threshold=2),
       stopping_criterion=AllCriteria(criteria=(
           LearningThreshold(),
           BetaBounds(consecutive=2),
           BetaStability(consecutive=2),
       )),
       seed=7,
   )
   result = analysis.run()

``Surrogate`` predicts mean/spread from row-wise independent standard normal
points. ``LearningFunction.select`` chooses an available candidate and returns
a ``LearningDecision``. ``StoppingCriterion`` inspects an immutable fit history
and separately accepts or rejects final sampling precision. Its policies are
stateless across runs. ``ReliabilityEstimator.estimate`` receives a batched
surrogate predictor and a run-owned random generator. It returns a
``ReliabilityEstimate`` identifying the sampling method, dependence and
uncertainty calculation. It cannot call the true limit state through this
interface. Weighted or correlated sampling must provide its own uncertainty
calculation; the runner does not recompute an IID binomial CoV.

The concise settings ``surrogate="kriging"``, ``learning_function="u"``,
``n_estimation`` and ``target_cov`` remain available for the default workflow.
An explicit component cannot be combined with its shortcut settings: configure
sample size on the estimator, threshold on the learning function and sampling
precision on the stopping policy. This avoids silently ignored options.

``result.history`` records the exploratory probability and the probabilities
obtained by classifying mean + 2 std and mean - 2 std. The fixed MC workflow
uses sample proportions; adaptive estimators supply their own measure-correct
probabilities. Transforming these
endpoints gives an ascending ``beta_band``. ``BetaBounds`` and
``BetaStability`` follow Eqs. (2) and (3) of [Moustapha2022]_, including default
three-consecutive-test requirements. Setting both to two consecutive tests
reproduces the review's combined stopping rule; the example above adds the
learning threshold as a third requirement. At beta zero or unresolved infinite
endpoints, the relevant beta test cannot pass. For negative beta the relative
denominator uses its absolute value.

These are **surrogate sensitivity diagnostics**, not confidence intervals for
the true failure probability. Bootstrap spread in particular need not be
Gaussian and can miss common model bias. A stable but inaccurate surrogate can
satisfy a beta-stability test. ``result.estimate`` keeps final sampling
diagnostics separate from these bands, and every default policy still requires
adequate final sampling precision before reporting convergence.

Active Kriging with subset simulation
-------------------------------------

Use ``estimator=SubsetSimulationEstimator()`` to resample nested conditional
populations after every surrogate fit, using the nested-event construction
of [AuBeck2001]_. Its default settings are 2000 samples
per level, intermediate probability 0.1, four independent replications, pCN
proposal scale 0.5 and at most 12 levels. These are practical starting settings,
not the review's much larger benchmark configuration. Set ``n_samples`` and
``n_replications`` on the estimator to improve sampling precision;
``n_candidates`` only caps the unique points passed to learning-point selection.
The default initial design increases to ``max(30, 5*n_variables)`` for adaptive
estimators. A larger design helps establish disconnected failure surfaces.

Exploration uses **mean - 2 std** to build the subsets. Until the terminal
level, every conditioning threshold is positive. Failure under mean + 2 std,
mean and mean - 2 std is therefore nested inside the same conditioning event.
The three probabilities use the same product of intermediate conditional
probabilities and the same final conditional population. This preserves their
ordering without treating pooled conditional samples as draws from the prior.
All generated level states are eligible for the enrichment pool, with duplicate
states removed and uniform subsampling if the pool cap is exceeded. Previously
observed points cannot be evaluated again. This is a documented optimistic-band
variant of the modular framework in [Moustapha2022]_, not a reproduction of all
its benchmark settings or of Bayesian subset simulation.

The Gaussian-preserving pCN proposal [CotterEtAl2013]_ is reversible for the
independent standard normal prior. Acceptance only tests membership in the
current subset. Initial seeds are included in each chain, and ties in an
adaptive threshold retain their actual conditional fraction. Nondecreasing
positive thresholds terminate with ``stalled``; exceeding the level budget
returns ``max_levels``. Incomplete exploration cannot satisfy the stopping
policy, but the runner can continue adding observations to improve the
surrogate. Independent final-estimator failure produces ``estimation_failed``
when surrogate stopping had otherwise succeeded.

A separate exploration seed is restarted after each fit, providing common
random numbers for convergence comparisons. The final estimate starts fresh
replications on the frozen surrogate **mean**, independently of enrichment.
Each ``SubsetRun`` records levels, acceptance rates, probability and a
within-chain CoV approximation. ``result.estimate.sampling_cov`` uses the
larger standard error from independent full-run replication and the aggregate
within-chain approximation. Replication reflects variation from chain ancestry
and adaptive levels that an IID calculation omits. This is still an estimated
sampling error, particularly noisy with few replications; it does not bound
bias or guarantee discovery of all failure regions. No binomial interval is
assigned to subset samples.

The :doc:`notebooks/ex_active_subset` tutorial demonstrates rare and disconnected
failure events, diagnostics, independent validation and visible budget failure.
The default U-threshold stopping policy is available, but combined beta-band
and beta-stability criteria often avoid continued enrichment of remote points
whose contribution to failure probability is negligible.

Next implementation increments
------------------------------

The component foundation above is implemented. Continue the v2.0 sequence with:

1. Add **PC-Kriging**, combining a selected polynomial trend and a Gaussian
   process residual with the corresponding predictive uncertainty. Validate
   against original reference implementations and published problems.
2. Add **FBR learning** for bootstrap PCE, extending the prediction/selection
   contract to retain replicate classifications rather than inferring votes
   from a mean and standard deviation. Include premature-stop cases.
3. Add an **active-learning importance-sampling** workflow where its proposal
   assumptions are appropriate; distinguish a design-point-centered proposal
   from methods that can discover multiple separated failure regions.

An adaptive quadratic response-surface workflow is a useful later candidate
for classical structural-assessment practice. SVM-based reliability can remain
an optional later extension: a classification margin is not a predictive
standard deviation and needs its own learning/stopping contract. A generic
neural-network or general UQ suite is outside this scope.

Acceptance evidence
-------------------

Use the same published/analytic problems and multiple seeds across methods.
The suite includes normal-sum, lognormal-beam and four-branch benchmarks,
rare linear and two-tail active problems, and a replicated subset check of an
analytic probability near 2.9e-7 in both 2 and 20 dimensions. Measure
probability error, missed-region/classification error, true model evaluations,
sampling error and explicit nonconvergence. These cases do not reproduce the full 20-problem review benchmark.

See :doc:`notebooks/ex_active_learning` for implemented methods, numerical
references and limitations, and :doc:`theory` for their current formulation.
