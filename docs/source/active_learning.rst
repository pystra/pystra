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
     - Explicit final-estimator interface; MonteCarloEstimator with
       independent final sampling; fixed MC enrichment pool
     - Active-learning integration with subset simulation and importance sampling
   * - Learning functions
     - Explicit selection interface; U and EFF
     - Bootstrap voting/fraction of bootstrap replicates (FBR); batch selection
   * - Stopping
     - Explicit policies for learning thresholds, beta bands, beta stability
       and combined criteria; separate final sampling precision;
       explicit budget/exhaustion status
     - Diagnostics driven by weighted/dependent reliability-estimator samples

Standalone FORM, SORM, crude Monte Carlo, importance sampling, line sampling
and subset simulation already exist. Having these solvers does not mean that
active learning can currently compose them. ``ActiveLearning`` now accepts
explicit final-estimation and stopping components. Its enrichment pool remains
fixed independent normal MC: substituting a final estimator does not by itself
implement an adaptive subset-simulation or importance-sampling algorithm.

The current Kriging/U loop is an AK-MCS-style implementation. EFF is provided
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

``result.history`` records the candidate probability and the probabilities
obtained by classifying mean + 2 std and mean - 2 std. Transforming these
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

Next implementation increments
------------------------------

The component foundation above is implemented. Continue the v2.0 sequence with:

1. Integrate **active Kriging with subset simulation** for rare events. Retain
   the standalone subset solver as a baseline and preserve dependence-aware
   sampling uncertainty. Replace the fixed enrichment pool with samples from
   the estimator, carrying its sampling measure through probability and beta
   diagnostics. A frozen-surrogate subset estimate alone does not
   constitute the complete adaptive algorithm.
2. Add **PC-Kriging**, combining a selected polynomial trend and a Gaussian
   process residual with the corresponding predictive uncertainty. Validate
   against original reference implementations and published problems.
3. Add **FBR learning** for bootstrap PCE, extending the prediction/selection
   contract to retain replicate classifications rather than inferring votes
   from a mean and standard deviation. Include premature-stop cases.
4. Add an **active-learning importance-sampling** workflow where its proposal
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
Extend the current normal-sum, lognormal-beam and four-branch benchmarks with
rare events, separated failure regions and moderate/high dimension. Measure
probability error, missed-region/classification error, true model evaluations,
sampling error and explicit nonconvergence. The existing three problems alone
do not reproduce the 20-problem review benchmark.

See :doc:`notebooks/ex_active_learning` for implemented methods, numerical
references and limitations, and :doc:`theory` for their current formulation.
