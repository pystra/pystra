# PySTRA 1.6.0 → 2.0 migration trials

Recorded 13 September 2026 on `v2.0-contracts`, with the actual revision and
library source hash retained in the generated artifact. These representative
user trials support the direct 2.0 release; the separate performance benchmark
and method-validation tests cover different release gates.

**Result:** all seven 1.6.0 example scripts and eleven tutorial notebooks passed
conversion/idempotence checks. Eleven complete scripts/tutorials were manually
migrated, executed in both versions and passed their numerical comparisons.
The sandbox run could not initialize OpenSees MPI. Claude subsequently ran all
twelve trials in the maintainer environment on integration revision `609cba7`;
all twelve passed, including OpenSees under both versions.

## Reproduce the trials

The baseline is the exact source at `v1.6.0`. Shared Git metadata is read-only
in this session, so `git archive` exported the tag instead of creating a detached
worktree. Neither package was installed over the other; every subprocess sets
`PYTHONPATH` to its selected checkout's `src`. The observer records the imported
`pystra.__file__` and version with its results.

```sh
mkdir -p /tmp/pystra-v1.6.0-trials
git archive v1.6.0 | tar -x -C /tmp/pystra-v1.6.0-trials
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pystra-mpl \
  python docs/migration/trial_scripts/run_trials.py \
  --baseline /tmp/pystra-v1.6.0-trials --output /tmp/pystra-migration-trials
```

The runner exits nonzero for an execution or numerical comparison failure.
`--skip-external` explicitly omits OpenSees execution while retaining its
conversion check. It writes the extracted baseline scripts, converter outputs,
reviewed manual diffs, logs, numerical records and comparison results.

The environment was Python 3.13.12, NumPy 2.4.4 and SciPy 1.17.1, with one
OpenMP/BLAS thread and no pytest-xdist. Figures use the noninteractive Agg
backend. The trial observer wraps analysis `run()` methods only to record
completed results; it does not replace distributions or numerical algorithms.
The 1.6.0 global random stream starts at seed 20260913, with the simulation
notebook's own seeds retained. Migrated simulations receive explicit integer
`rng` seeds; their default NumPy Generators intentionally have different streams.

[Machine-readable evidence](trial_scripts/results.json) contains the complete
conversion inventory, comparison criteria and numerical records.
The runner writes this combined artifact directly as `OUTPUT/results.json`;
copy it to `docs/migration/trial_scripts/results.json` after reviewing a new run.
Revision IDs, library source hashes and dependency versions are captured at
execution time. Source locations use `<repo>` and `<baseline>` placeholders;
no hand-written environment label or separate merge step is needed. Reruns
preserve the schema and numerical criteria; revisions and environment values
reflect the actual checkouts used.
[Reviewed migrated scripts](trial_scripts/) and their
[manual patches](trial_scripts/manual/) are retained alongside the runner.

## What was converted automatically

`python -m pystra.migrate PATH...` defaults to unified diffs. `--write` applies
them. It changes resolved PySTRA imports/module paths, module attributes such
as `ra.Form`, and known constructor keywords such as `stochastic_model`,
`analysis_options`, `stdv` and `startpoint`. A renamed direct import retains its
local binding (`from pystra import FORM as Form`). User identifiers, strings
and comments are preserved. Every conversion checks a second pass for changes.

Notebook imports carry across valid Python code cells in document order.
Markdown, outputs and metadata are preserved. IPython syntax, wildcard imports,
ambiguous bindings, split module imports and positional configuration needing
judgment are left for review. Constructor, options, getter, printing,
calibration and random-seeding diagnostics are not claims of runnable code.
The installed mapping is generated from `api-migration.json`, and
`scripts/generate_migration_data.py --check` verifies agreement.

All 18 original files were tested. Besides the numerical trials below, the
conversion-only set includes `example_parallel_multithreading.py`,
`ex_code_calibration.ipynb`, `ex_generic_calibration.ipynb`,
`ex_load_combinations.ipynb`, `ex_timing.ipynb` and
`example_global_calibration.ipynb`. The old multithreading example's wildcard
import is deliberately left unchanged. The other calibration/workflow
notebooks retain manual-review diagnostics; the full factor-calibration
notebook represents that migration in the numerical gate.

## Reviewed manual edits

* Model instance methods become `add_variable` and `set_correlation`.
* Mutable `AnalysisOptions` and its setters become the appropriate
  `FORMOptions`, `SORMOptions` or `SimulationOptions`. The DDM examples set
  `FORMOptions(differentiation=...)`. Point-fit SORM uses
  `SORMOptions(fit="point")` before `run()`.
* Each explicit `run()` returns a named record. Getters and printing use
  `beta`, `failure_probability`, `n_limit_state_evaluations`, diagnostics and
  `summary()` on that record. The DDM tutorial returns the record from its
  helper function rather than returning a solver.
* The introductory tutorial's positional Uniform parameter mode becomes
  explicit `lower=` and `upper=` keywords. Engineering limit-state expressions
  and distribution values are preserved.
* Simulations use explicit `rng` seeds and sample budgets. The simulation
  tutorial reuses the exact model and limit-state specification belonging to
  its supplied FORM solver; 2.0 rejects a solver tied to a different model.
* Sensitivity method/delta selection moves to construction. Result access uses
  `.marginal` and `.correlation`. The numerical sensitivity implementation
  explicitly retains the old perturbed-mean start-point convention while the
  public distribution-copy contract preserves start points by default.
* The factor-calibration tutorial uses `LoadCombination.from_actions`,
  `FactorCalibrationProblem`, `solve_designs`, `derive_factors`,
  `select_factors`, `design_with_factors` and `verify_designs`. Its old implicit
  factor-selection policy is explicit: minimum resistance factors, maximum
  load factors and maximum companion combination factors. All four original
  coefficient/matrix and root/alpha studies are retained, including the
  nonlinear resistance/load-error example.

Two fixes are necessary even before migrating the released examples in this
environment. `examples/sensitivity.py` needs `form.run()` before printing its
results. In `ex_simulation.ipynb`, one-element getter arrays use
`float(np.asarray(value).item())` because NumPy 2.4 rejects their direct
conversion with `float(value)`. These changes affect neither input model nor
numerical method. Timing demonstrations run one repetition in both versions;
wall-clock speed is not part of this evidence.

## Numerical comparisons

FORM values use absolute/relative tolerance `1e-10`. Marginal sensitivities use
absolute tolerance `1e-9` and relative tolerance `1e-8`. Calibration case designs
and verified indices use absolute tolerance `1e-8`. SORM allows `2e-4` in beta
and `0.1%` relative failure probability, covering the existing 2.0
finite-difference curvature differences from 1.6.0. C4 does not alter those
curvature kernels.

| Original script/tutorial | Representative result, 1.6.0 | Migrated 2.0 | Outcome |
| --- | ---: | ---: | --- |
| `examples/example.py` | FORM beta 3.734766588353375 | 3.734766588353376 | Pass; SORM, CMC, IS and distribution analysis also run |
| `examples/ddm_example.py` | FFD/DDM beta 1.621684841633 / 1.621684998130 | Same to shown precision | Pass |
| `examples/gev_example.py` | FORM beta 1.961004662500 | 1.961004662500 | Pass; SORM also compared |
| `examples/sensitivity.py` | beta 2.121787605371; numerical derivatives | Same; derivatives agree | Pass |
| `examples/timing.py` | Built-in/SciPy SORM beta 1.848997969 | 1.849133185 | Pass within SORM tolerance |
| `ex_intro.ipynb` | FORM beta 1.753976140741; point-fit SORM 1.833488678688 | 1.753976140741 / 1.833488676615 | Pass; curve-fit SORM and simulations also compared |
| `ex_ddm.ipynb` | FFD/DDM beta 1.621684841633 / 1.621684998130 | Same to shown precision | Pass |
| `ex_scipy_distributions.ipynb` | FORM beta 1.961004662500 | 1.961004662500 | Pass; SORM also compared |
| `ex_sensitivity.ipynb` | Closed-form and numerical derivatives, two models and four step sizes | Agree within declared tolerances | Pass |
| `ex_simulation.ipynb` | Sphere/parabola FORM beta 5.000337015053 / 2.999999963213 | Same to shown precision | Pass; six simulation comparisons below |
| `ex_factor_calibration.ipynb` | Four governing design scales 3.047717273919 / 3.047717013955 / 3.047709596618 / 1.298049802300 | Same to shown precision | Pass; all eight case designs and eight verification betas differ by less than `2e-12` |

Independent stochastic streams are compared using the difference in estimates
divided by their pooled reported standard error:
`sqrt((p1 * cov1)**2 + (p2 * cov2)**2)`. The trial criterion is at most five
pooled standard errors; it is a regression-screening tolerance, not a guarantee
that either estimate is precise. The original sample budgets are retained.

| Simulation tutorial case | 1.6.0 failure probability | 2.0 failure probability | Pooled standard errors |
| --- | ---: | ---: | ---: |
| Sphere, CMC | 0.005400 | 0.005300 | 0.14 |
| Sphere, line sampling | 0.000992389 | 0.005945021 | 0.99 |
| Sphere, subset simulation | 0.005130 | 0.004695 | 0.64 |
| Parabola, CMC | 0.006200 | 0.005900 | 0.39 |
| Parabola, line sampling | 0.006649044 | 0.006180413 | 0.84 |
| Parabola, subset simulation | 0.005130 | 0.008945 | 3.87 |

The sphere line-sampling trial has only 200 lines and a 2.0 CoV of about 0.84;
its broad uncertainty and the pre-existing 2.0 two-sided line-root improvements
make a comparison of printed point estimates alone misleading. The sphere's
independent reference is `chi2.sf(25, 10) = 0.0053455055`. This trial records the
limited precision instead of treating migration as validation of the method.
The example/intro CMC and importance-sampling differences range from 1.09 to
1.57 pooled standard errors.

## External environment and remaining release checks

`examples/openseespy_ex.py` passed under both versions in the maintainer
environment on 13 September 2026. Claude ran the complete runner without
`--skip-external` on integration revision `609cba7`, against a fresh archive of
`v1.6.0`: exit status 0, all twelve trials `passed`. That environment used
Python 3.13.12, OpenSeesPy 3.8.0.0, opsvis 1.3.7, NumPy 2.4.4, SciPy 1.17.1,
pandas 3.0.2 and Matplotlib 3.10.8, with one OpenMP/BLAS thread and `Agg`.

The committed machine-readable artifact records the independently regenerated
sandbox run: eleven numerical passes and OpenSees execution failures with
status 15 because `MPI_Init` cannot create a listener socket (`Operation not
permitted`). It does not relabel that failure as the maintainer pass. Use
`--skip-external` for sandboxed reproductions; the complete migrated OpenSees
script remains available for unrestricted runs.

These trials do not replace the integration branch's complete notebook,
platform/dependency, installed-package, performance or external-solver release
gates. The trial runner makes failures visible and preserves the evidence
needed to repeat the comparisons on the final release commit.
