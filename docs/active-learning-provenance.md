# PC-Kriging, bootstrap voting and active importance sampling

PySTRA 2.0 targets **a carefully validated implementation of established and
selected modern structural reliability methods**, coupled with practical code
calibration and structural assessment tools. These additions implement selected
methods within that scope; they do not claim comprehensive state-of-the-art
coverage or endorsement by their authors.

## PC-Kriging

`PCKrigingSurrogate` implements sequential PC-Kriging: select a sparse Hermite
trend using the existing adaptive PCE implementation, then re-estimate its
coefficients by generalized least squares within a correlated residual model.
The conditional prediction variance includes uncertainty in those trend
coefficients. It does not integrate over kernel parameters or trend selection.

Correlation is Matérn 5/2 by default, consistent with `KrigingSurrogate`;
Gaussian correlation is an explicit option. Anisotropic length scales use
profiled maximum likelihood, an analytic gradient and bounded L-BFGS-B.
The objective is scaled per observation; failed optimization raises, and a
failed fit invalidates previous predictions. Fixed scales are supported for
numerical comparisons. This is sequential PC-Kriging, not the optimal variant
that evaluates Kriging models along every LAR support. The implementation
uses NumPy/SciPy and adds no dependency on a general UQ framework.

`tests/data/uqlab_active.json` compares fixed-trend, fixed-Gaussian-kernel GLS
coefficients, residual variance and universal predictions against **unmodified
UQLab 2.2.0 routines executed with GNU Octave 10.3.0**. The supplied trend
supports are fixture inputs: this comparison does not rerun the full UQLab
PCK framework or compare its optimizer. Sparse trend selection already has
separate executed UQLab comparisons in `uqlab-pce-provenance.md`. Independent
tests solve the constrained predictor through an augmented linear system,
check analytic likelihood gradients and verify the trend-variance correction.

The four-branch benchmark uses Matérn correlation, U >= 3, beta bounds and
beta stability, across three seeds. Gaussian correlation and beta stability
alone proved too confident for this nonsmooth response during development.
The successful benchmark settings are explicit, and independent checks cover
each failure region. They are not a guarantee of discovery on another model.

## Failed bootstrap replicates (FBR)

`PCESurrogate.predict_replicates` returns actual bootstrap responses with
consistent replicate columns across calls. `FBRLearning` minimizes
`abs(B_safe - B_failure) / B`; zero response counts as failure. No Gaussian
approximation to replicate votes is made. UQLab's RBDO helper returns the
negative score for maximization; our minimization convention reverses its
sign. The fixture checks this relationship.

`BootstrapBounds` applies the paper's min/max bootstrap probability range,
divided by the full-design probability estimate, for two consecutive fits
(default tolerance 0.1). These diagnostics use all points in the common IID
normal pool, including previously selected candidates. Final sampling is
independent. The range need not contain the full-design estimate and is not a
confidence interval for total error. BootstrapBounds is rejected for adaptive
weighted/conditional pools because those estimators do not yet provide
replicate-specific probability estimates. FBR selection itself can use those
pools with another explicit stopping policy.

The workflow follows the paper's single-point option. It retains PySTRA's LHS
initial design and selected-support bootstrap, without the paper's clustered
batch selection. Tests verify non-Gaussian votes, convergence on the lognormal
beam, explicit four-branch budget exhaustion, and a hidden-mode counterexample
where bootstrap unanimity does not establish accuracy.

## Active importance sampling

`ImportanceSamplingEstimator` composes an AK-IS-style workflow with explicit
Gaussian proposal centres in independent normal coordinates. One centre can
be a converged FORM design point; multiple supplied centres cover known
separated modes. A default 10% target-normal component bounds weights.
Setting the defensive fraction to zero gives ordinary shifted Gaussian-mixture
IS and requires adequate tail coverage and finite second moments for the
reported sampling-error approximation.

The surrogate is enriched on proposal points, and the probability/band
calculations use likelihood weights throughout. The proposal is fixed within
a run. Final sampling starts independently on the frozen surrogate mean.
This does not implement adaptive discovery of centres, a surrogate-dependent
optimal proposal, or a true-model correction factor. Counts exclude any FORM
analysis used to supply centres and any independent validation calls.

The ordinary estimator is `mean(I * f/q)`; weights are not self-normalized.
Its standard error comes from the variance of these weighted contributions.
Effective failure counts, mean/max weights and raw estimates are retained.
An out-of-range finite estimate is exposed at the nearest probability endpoint
with explicit invalid status and infinite CoV; the raw value stays visible.
Sensitivity bands alone are clipped to probability limits. Zero/all observed
failures or inadequate effective failure counts remain incomplete. No binomial
confidence interval is attached to a weighted estimate.

Independent tests compare ensemble bias and reported error against a normal
half-space probability near 2.9e-7 in 2 and 20 dimensions, and test weighted
bands, candidate caps and rare disconnected events. A defensive component and
healthy effective sample counts cannot prove that all relevant modes were found.

## Source notices and reproducibility

The local ignored UQLab installation is unchanged. Copyright (c) 2018–2026,
Stefano Marelli and Bruno Sudret (ETH Zurich). The full BSD-3-Clause notice is
retained in `THIRD_PARTY_NOTICES`; PySTRA remains GPL-3.0-or-later. These source
files informed the implementation or supplied executed comparisons:

| UQLab 2.2.0 path | SHA-256 |
| --- | --- |
| `modules/uq_model/builtin/uq_metamodel/Kriging/eval/uq_Kriging_eval.m` | `bec2b9d266102b68937332943ca8ff7cf036eaa180fab8dc30718767b616be8a` |
| `modules/uq_model/builtin/uq_metamodel/Kriging/calc/uq_Kriging_calc_auxMatrices.m` | `d5db69f7dfaf28a53ddbf24a1c7fbebe48f4691f19f5c6a3f1bdd6c8d416a201` |
| `modules/uq_model/builtin/uq_metamodel/Kriging/calc/uq_Kriging_calc_DiagOfCongruent.m` | `647bbf3b93dd83515915626467a69f3b60109ab73f50c199deef00e21ca5a4dc` |
| `modules/uq_analysis/builtin/uq_rbdo/Metamodels/uq_LF_FBR.m` | `5f4535cccae31c2a55a7074e5e9f52f77ce07c777bbf8eb47c3f31d625e246b0` |

Regenerate the fixture with MATLAB or Octave, from the repository root:

```matlab
addpath('scripts');
validate_uqlab_active('al/UQLab_Rel2.2.0', 'tests/data/uqlab_active.json');
```

The script supplies plain numerical model structs and explicit Hermite/kernel
adapters. It calls UQLab's prediction and FBR routines without initializing
the toolbox framework. CI reads the stored numerical results without requiring
MATLAB, Octave or UQLab.

## References

- Schöbi, R., Sudret, B. and Wiart, J. (2015). Polynomial-chaos-based Kriging.
  International Journal for Uncertainty Quantification 5(2), 171–193.
  https://doi.org/10.1615/Int.J.UncertaintyQuantification.2015012467
- Schöbi, R., Sudret, B. and Marelli, S. Rare event estimation using
  polynomial-chaos Kriging. https://doi.org/10.1061/AJRUA6.0000870
- Marelli, S. and Sudret, B. (2018). An active-learning algorithm that combines
  sparse polynomial chaos expansions and bootstrap for structural reliability
  analysis. https://doi.org/10.1016/j.strusafe.2018.06.003
- Echard, B., Gayton, N., Lemaire, M. and Relun, N. (2013). A combined
  Importance Sampling and Kriging reliability method for small failure
  probabilities with time-demanding numerical models.
  https://doi.org/10.1016/j.ress.2012.10.008
- Moustapha, M., Marelli, S. and Sudret, B. (2022). Active learning for structural
  reliability: survey, general framework and benchmark.
  https://doi.org/10.1016/j.strusafe.2021.102174
- Teixeira, R., Nogal, M. and O’Connor, A. (2021). Adaptive approaches in
  metamodel-based reliability analysis: A review.
  https://doi.org/10.1016/j.strusafe.2020.102019
