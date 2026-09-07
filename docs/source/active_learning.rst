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
     - Active learning with a Monte Carlo candidate pool and independent
       final Monte Carlo estimation
     - Active-learning integration with subset simulation and importance sampling
   * - Learning functions
     - U and EFF
     - Bootstrap voting/fraction of bootstrap replicates (FBR); batch selection
   * - Stopping
     - Candidate learning threshold, final conditional sampling CoV,
       explicit budget/exhaustion status
     - Surrogate-based probability/beta diagnostics; configurable stability
       and combined criteria

Standalone FORM, SORM, crude Monte Carlo, importance sampling, line sampling
and subset simulation already exist. Having these solvers does not mean that
active learning can currently compose them. In particular, ``ActiveLearning``
still fixes its estimator and stopping logic inside its runner.

The current Kriging/U loop is an AK-MCS-style implementation. EFF is provided
as a learning function; this is not a complete EGRA reproduction. Likewise,
the PCE implementation reproduces specified UQLab sparse-selection routines
and fast-bootstrap behavior, with documented numerical differences, but uses
PySTRA's single-point enrichment and stopping contract. Dense quadratic PCE
can span quadratic response surfaces in normal coordinates, but does not
supply a classical adaptive
response-surface reliability algorithm.

Recommended next increments
---------------------------

This is the proposed development sequence, not a list of implemented features:

1. Give the reliability estimator and stopping criterion explicit interfaces
   alongside the existing surrogate interface. Keep coordinates, event signs,
   sample weights, sampling dependence and uncertainty diagnostics explicit.
   Add surrogate-based probability/beta convergence diagnostics without
   presenting them as rigorous bounds on the true failure probability.
2. Integrate **active Kriging with subset simulation** for rare events. Retain
   the standalone subset solver as a baseline and preserve dependence-aware
   sampling uncertainty. A frozen-surrogate subset estimate alone does not
   constitute the complete adaptive algorithm.
3. Add **PC-Kriging**, combining a selected polynomial trend and a Gaussian
   process residual with the corresponding predictive uncertainty. Validate
   against original reference implementations and published problems.
4. Add **FBR learning** for bootstrap PCE and a small set of documented stopping
   policies. Include both successful and premature-stop cases.
5. Add an **active-learning importance-sampling** workflow where its proposal
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
