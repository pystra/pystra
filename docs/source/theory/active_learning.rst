Active-learning reliability
***************************

Active learning reliability
===========================


``pystra.active_learning.ActiveLearning`` combines a surrogate with Monte
Carlo classification and sequential true limit-state evaluations. Kriging
follows the AK-MCS approach [Echard2011]_. The separation of surrogate,
reliability estimator, learning function and stopping criterion follows the
framework discussed by [Moustapha2022]_. The complementary review
[TeixeiraNogalOConnor2021]_ surveys the main adaptive metamodel families.
See :doc:`/active_learning` for implemented choices and limitations, and the
:doc:`/notebooks/ex_active_learning` tutorial for independent benchmark references.

An initial Latin hypercube design and a fixed normal Monte Carlo candidate
pool are constructed in **independent standard normal coordinates**. The
configured Nataf or Rosenblatt transformation maps only true evaluations to
physical space. Thus Hermite orthogonality is with respect to independent
normals, including when physical marginals are nonnormal or dependent.
A Student-t spherical Nataf space is unsupported; select Rosenblatt instead.

The surrogate is refitted after each enrichment. Previously evaluated
candidates cannot be selected again. The default initial size is
``max(12, 2*n_variables)``; sparse PCE uses ``max(30, 5*n_variables)``.
Dense OLS uses twice its largest total-degree basis size by default.
By default, the final probability estimate uses an independent Monte Carlo
population that never participates in fitting or point selection. An explicit
``ReliabilityEstimator`` can replace final estimation and owns its sampling
uncertainty calculation. ``EnrichmentEstimator`` implementations additionally
supply new candidate populations and probability diagnostics after each fit.
``SubsetSimulationEstimator`` uses this contract for active subset simulation.
``ImportanceSamplingEstimator`` provides Gaussian-mixture importance sampling
with explicit centers and weighted probability diagnostics. See
:doc:`/notebooks/ex_active_extensions` for its coverage and checks.

Surrogates and uncertainty
--------------------------

Kriging uses scikit-learn's Matérn 5/2 Gaussian process with response
normalization and a small numerical nugget. Install the optional ``al`` extra.
Optimizer convergence warnings remain visible; they concern hyperparameter
fitting, separately from the reliability stopping status.

``PCESurrogate`` uses normalized probabilists' Hermite polynomials, selecting
sparse terms by hybrid least-angle regression [BlatmanSudret2011]_. The default
candidate degrees are 1 through 5. ``degree`` and ``q_norm`` can each specify
an increasing sequence: every candidate is fitted and the best corrected
leave-one-out error retained. Hyperbolic truncation and ``max_interaction``
limit the candidate dictionary; ``max_terms`` guards against excessive size.

The implementation adapts the local UQLab 2.2.0 routines, with the copyright
and BSD terms retained in ``THIRD_PARTY_NOTICES``. Direct numerical regression
fixtures compare the original UQLab routines under Octave with PySTRA.
SVD solves replace normal-equation inverses; bootstrap indices are sampled
uniformly. ``method="ols"`` retains dense least-squares fitting.

UQLab's centered, normalized path scoring selects a sparse support. A final
OLS fit on the original Hermite columns supplies the mean and corrected LOO
score used to compare degrees/truncations. ``fit_result`` exposes the selected
degree, q-norm, indices, coefficients and candidate error diagnostics.
Optional early stopping can miss an isolated higher-order term: set
``degree_early_stop=False`` and ``q_norm_early_stop=False`` for exhaustive search.

Pairs-bootstrap refits of the **selected sparse support** supply local spread,
following the fast-bootstrap approach of [MarelliSudret2018]_. Selection is
repeated at each enrichment, but held fixed within each bootstrap ensemble.
The reliability loop still enriches one point at a time; batch enrichment and
full bootstrap model reselection are separate extensions. Rank-deficient
bootstrap draws use minimum-norm least squares, as in UQLab, and their count
is exposed in ``fit_result.n_rank_deficient_bootstrap``. This makes a weak
resampled design visible without conditioning the bootstrap on full rank.

Bootstrap spread is not a Gaussian posterior, a calibrated confidence band,
or a bound on polynomial truncation bias. A common bias across all bootstrap
fits can produce confidently wrong classifications, particularly for nonsmooth
series-system surfaces. Use independent true evaluations, polynomial-degree
checks and benchmark comparisons before trusting a PCE reliability estimate.

Learning functions
------------------

The U-function [Echard2011]_ selects the smallest value of
:math:`U=|\mu|/\sigma`, stopping when its minimum reaches the configurable
threshold (default 2). At zero spread, U is infinite away from the boundary
and zero on it.

The expected feasibility function [Bichon2008]_ selects the largest

.. math::

   \mathrm{EFF} = E[\max(0,\varepsilon-|G|)],
   \qquad G\sim N(\mu,\sigma^2),\quad \varepsilon=2\sigma.

This expectation is symmetric in the mean and nonnegative. Its default
stopping tolerance is :math:`10^{-3}` in **limit-state units**, so rescaling
the limit state requires rescaling this tolerance. Zero spread gives zero
EFF. With PCE, the Gaussian assumption is a heuristic applied to bootstrap
spread; it does not convert that spread into a posterior distribution.

Stopping and interpretation
---------------------------

The default ``LearningThreshold`` stops enrichment when the configured score
threshold is met and the candidate pool contains both predicted failure and
survival. Selection is independent of this decision: a ``LearningFunction``
returns a selected index, score and threshold flag, while a
``StoppingCriterion`` inspects the complete fit history.

Every fit records two candidate proportions:

.. math::

   p_{\mathrm{lo}} = \frac{1}{N_c}\sum_j
       \mathbf{1}\{\mu_j+2\sigma_j\leq0\},\qquad
   p_{\mathrm{hi}} = \frac{1}{N_c}\sum_j
       \mathbf{1}\{\mu_j-2\sigma_j\leq0\}.

The corresponding ascending beta band is
:math:`[\beta_{\mathrm{lo}},\beta_{\mathrm{hi}}]
=[-\Phi^{-1}(p_{\mathrm{hi}}),-\Phi^{-1}(p_{\mathrm{lo}})]`.
``BetaBounds`` uses
:math:`(\beta_{\mathrm{hi}}-\beta_{\mathrm{lo}})/|\hat\beta|\leq0.01`
for three consecutive fits. ``BetaStability`` instead requires
:math:`|\hat\beta_i-\hat\beta_{i-1}|/|\hat\beta_i|\leq0.005`
for three consecutive changes (at least four fits). These are the rules in
Eqs. (2)-(3) of [Moustapha2022]_, with absolute denominators to also handle
negative beta. Nonfinite required indices and a zero denominator do not pass.
``AllCriteria`` requires all supplied policies; setting both beta policies to
two consecutive tests gives the review's combined rule. Learning-threshold
acceptance can be included as another requirement.

These bands measure the sensitivity of classifications to surrogate spread.
They are not rigorous bounds on the true probability, simultaneous Gaussian
confidence bands, or a correction for model bias. In particular, stable biased
predictions can pass a beta-stability test. The formulas above apply to the fixed
unweighted candidate pool. Active subset simulation instead builds nested
positive-threshold events from mean - 2 std; all three target failure events
are subsets of the same final conditioning event. Their probabilities use the
product of intermediate conditional probabilities times the corresponding
final conditional fraction. Pooled level states serve only for enrichment.
See :doc:`/active_learning` for that construction, the pCN sampling kernel and
replication-based uncertainty diagnostics.

An evaluation budget or exhausted pool returns explicit nonconvergence. The
final sample must separately meet the stopping policy's ``target_cov``
(default 0.1); otherwise the status is ``sampling_precision``. Zero or all
failures never pass the built-in precision checks. Estimators supply their own
CoV and optional confidence interval, including any corrections needed for
sample dependence; the runner does not impose a binomial formula on them.

The immutable result contains the estimate, normal-equivalent beta,
convergence status, true evaluation count, history, conditional sampling CoV
and, for default Monte Carlo, an exact 95% binomial interval. These sampling diagnostics exclude
surrogate error. A successful stopping status concerns the sampled points;
it cannot guarantee discovery of disconnected failure regions or eliminate
surrogate bias. Nonconvergence emits a warning and preserves an explicitly
unfinished estimate for diagnosis.


Replicated subset sampling uncertainty
--------------------------------------

For conditional level j, let p_j be the indicator mean and S_c the sum of
centered indicators in chain c. The chain-cluster variance approximation is

.. math::

   v_j = \max\left(\frac{p_j(1-p_j)}{N_j},
          \frac{C_j}{C_j-1}\frac{\sum_c S_c^2}{N_j^2}\right).

The initial independent level uses only the Bernoulli term. For one subset
run, :math:`\delta_r^2=\sum_j v_j/p_j^2` gives an approximate squared CoV.
This accounts for within-chain clustering but omits cross-level dependence
and shared ancestry between different chains. ``variance_factor`` reports
the ratio of this level variance to its IID value.

For R independent complete subset runs with estimates q_r, the final estimate
is their mean. The reported standard error uses

.. math::

   s_{\mathrm{between}}^2 =
       \frac{\sum_r(q_r-\bar q)^2}{R(R-1)},\qquad
   s_{\mathrm{within}}^2 = \frac{\sum_r(q_r\delta_r)^2}{R^2},\qquad
   \mathrm{CoV} = \frac{\max(s_{\mathrm{between}},s_{\mathrm{within}})}{\bar q}.

The replication term captures variation of the complete adaptive sampling
procedure. The within-chain term prevents spurious precision when a small
set of replications happens to agree. Neither is a confidence bound, nor do
they remove the finite-sample bias of adaptive thresholds or surrogate error.
Incomplete runs, zero estimated probabilities in any replication, and endpoint
aggregate probabilities yield infinite CoV. No binomial interval is reported.

**Use this method:** :doc:`/active_learning` · :doc:`/notebooks/ex_active_learning` · :doc:`/api/active_learning`

For coordinate conventions, see :doc:`notation`.
