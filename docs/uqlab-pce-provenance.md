# UQLab source basis for PySTRA adaptive sparse PCE

PySTRA adapts the local **UQLab 2.2.0** sparse-PCE algorithm for independent
normal coordinates. Copyright (c) 2018–2026, Stefano Marelli and Bruno Sudret
(ETH Zurich). The complete BSD notice is retained in `THIRD_PARTY_NOTICES`,
which is included in both source and wheel distributions alongside PySTRA's
GPL-3.0-or-later licence. No endorsement is implied.

The installation inspected is `al/UQLab_Rel2.2.0` (ignored by Git). It remains
unchanged. The following files informed the implementation; hashes identify
the exact local source rather than relying solely on the release-directory name.

| Source relative to UQLab root | SHA-256 |
| --- | --- |
| `lib/uq_regression/LAR/uq_lar.m` | `b26d9c7d59ce3b2271a692b8b5dea5d99b3a54cd095ddc20e17826151c18782d` |
| `modules/uq_model/builtin/uq_metamodel/PCE/uq_PCE_loo_error.m` | `0a683ef5ce62fd789110c41a0d4f8630e864b4578e2ddea6a6f12a09e3b9c7b0` |
| `modules/uq_model/builtin/uq_metamodel/PCE/PolyCoeff/Regression/uq_PCE_calculate_coefficients_regression.m` | `92591c3b22c5a82c1ead24551201b3b8b8c3422cee4530a86785e5132cd7786a` |
| `modules/uq_model/builtin/uq_metamodel/PCE/PolyCoeff/Regression/uq_OLS_bootstrap.m` | `f5fb9e4dc4a57d1a2fbe1cbe1c43956c22a1be6018a66f826a7bdf34cfd31cd2` |
| `modules/uq_model/builtin/uq_metamodel/PCE/PolyBasis/uq_generate_basis_Apmj.m` | `7652c5fb02765917b38af5d3d9b7f60c36af54cd1d4f4f1e6a94479356748d2f` |
| `lib/uq_bootstrap/uq_bootstrap.m` | `17878728e62685d6d9eaa706b61d0a27d739d8ee576545c30fc8b01204a7b3fe` |

## Numerical mapping

- `uq_lar` / `uq_PCE_lars`: center and normalize nonconstant regressors;
  follow the least-angle path; score active supports using corrected LOO;
  refit the selected support, including its constant term, by OLS.
- `uq_PCE_loo_error`: PRESS residuals divided by population response variance,
  with correction `N/(N-P) * (1 + trace((Psi.T @ Psi)^-1))`. Centered path
  scoring adds intercept leverage `1/N` while counting nonconstant terms;
  final uncentered OLS scoring includes the constant term. These two scores
  must not be conflated.
- `uq_PCE_calculate_coefficients_regression`: compare final corrected OLS
  errors across degree/q-norm candidates, retain the best, and optionally
  stop after two unsuccessful candidates. Refit from scratch at enrichment.
- `uq_generate_basis_Apmj`: total-degree, hyperbolic and maximum-interaction
  truncations. Candidate ordering follows degree, interaction order,
  integer partition and lexicographic permutation.
- `uq_OLS_bootstrap`: freeze the selected sparse support and resample training
  pairs to construct the local spread. This is fast bootstrap, not a full
  repetition of adaptive model selection inside each bootstrap replicate.

## Intentional differences

PySTRA uses SVD-based solves rather than normal-equation/block-inverse updates,
rejects unidentifiable supports, treats residuals at floating-point roundoff as
zero, and limits candidate dictionary size. Bootstrap draws with insufficient
rank use minimum-norm least squares, as in UQLab; their count is exposed in
`fit_result.n_rank_deficient_bootstrap`. No resamples are discarded. Invalid fits clear previous results.
There are no weighted-output regressions or non-Hermite bases in this API:
PySTRA's probability transformation supplies independent normal coordinates.
All random sampling belongs to a local generator.

The local `uq_bootstrap.m` uses `round(rand(B,N)*(N-1))+1`. For N>1 this gives
each endpoint half the probability of an interior index. PySTRA uses uniform
integer draws. Direct bootstrap comparisons therefore use the same explicitly
stored **uniform** indices on both sides; they do not reproduce this weighting.

Degree/q-norm searches can be exhaustive. The implementation does not port
UQLab's LAR-path early-stop heuristic or target-accuracy option. The existing
PySTRA reliability loop still enriches one point at a time and uses its explicit
learning/final-sampling stopping contract. This is not a claim to reproduce every
strategy in the Marelli–Sudret paper or UQLab's full active-learning framework.

## Executed comparisons

`tests/data/uqlab_sparse_pce.json` stores inputs and numerical outputs from the
**unmodified UQLab routines executed with GNU Octave 10.3.0**, not from PySTRA.
MATLAB was not available on PATH. Octave was installed only in the temporary
`/tmp/pystra-uqlab-octave` environment. The tested routines are numerical MATLAB
functions; the full UQLab object framework was not initialized.

The cases cover an exact sparse polynomial, noisy polynomial, exponential
response, and the four-branch minimum. Tests compare degree, q-norm, selected
powers, coefficients, ordinary/corrected LOO, predictions and bootstrap spread.
The regenerated fixture uses exhaustive degree/q searches with LAR early stop
disabled. Relative LOO tolerance is 2e-6 because UQLab's normal equations lose
some precision on the high-degree exponential design; predictions/spread agree
within 2e-10 absolute tolerance. Fixture generation is independent of PySTRA.

Regenerate with MATLAB or Octave, from the repository root:

```matlab
addpath('scripts');
validate_uqlab_pce('al/UQLab_Rel2.2.0', 'tests/data/uqlab_sparse_pce.json');
```

Normal CI reads the fixture without requiring UQLab, MATLAB or Octave. Additional
tests compare PRESS with explicit deleted-row fits, recover an exact sparse
polynomial from an underdetermined dictionary, exercise adaptation/state reset,
and check independent true classifications in reliability benchmarks.

## References

- Blatman, G. and Sudret, B. (2011). Adaptive sparse polynomial chaos expansion
  based on Least Angle Regression. Journal of Computational Physics 230,
  2345–2367. https://doi.org/10.1016/j.jcp.2010.12.021
- Marelli, S. and Sudret, B. (2018). An active-learning algorithm that combines
  sparse polynomial chaos expansions and bootstrap for structural reliability
  analysis. Structural Safety 75, 67–74. https://doi.org/10.1016/j.strusafe.2018.06.003
- Moustapha, M., Marelli, S. and Sudret, B. (2022). Active learning for structural
  reliability: survey, general framework and benchmark. Structural Safety 96,
  102174. https://doi.org/10.1016/j.strusafe.2021.102174
