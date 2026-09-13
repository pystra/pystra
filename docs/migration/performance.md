# Performance against PySTRA 1.6.0

Measured September 13, 2026, before and after the C6e-perf optimizations on
`v2.0-contracts`. The reference is the exact `v1.6.0` export used for the
[migration trials](trials.md). The [initial evidence](performance-before-results.json)
and [optimized evidence](performance-results.json) retain each run's actual
revision, library source hash, dependency versions and portable import paths.
Both benchmarks include uncommitted library contents identified by those hashes;
the revision alone does not identify the optimized code.

**Outcome:** all six optimized workloads satisfy the 10% runtime threshold
against 1.6.0. Evaluation counts are unchanged and every numerical comparison
passes. All transformation-preparation medians also satisfy the threshold.
The additional simulation log-probability history remains a documented memory
cost; no tail accuracy, validation or failure diagnostics were removed.

## Reproduce and interpret the measurements

```sh
mkdir -p /tmp/pystra-v1.6.0-trials
git archive v1.6.0 | tar -x -C /tmp/pystra-v1.6.0-trials
python scripts/benchmark_against_1x.py \
  --baseline /tmp/pystra-v1.6.0-trials \
  --output /tmp/pystra-performance \
  --rounds 7 --samples 20000 --cpu 31
```

Choose an allowed CPU on another host, or omit `--cpu` to use its ordinary
scheduler. The output directory must be new. The runner uses one interpreter
with separate `PYTHONPATH` values, asserting that each imported PySTRA comes
from the requested checkout. It saves raw timing rounds, profiles, allocation
measurements and a generated `results.json`. The committed results are a copy
of that summary, without hand-edited measurements. Exported paths use
`<repo>`, `<baseline>`, `<output>` and interpreter placeholders; the
committed JSON was checked byte-for-byte against the generated file. `--summarize` recomputes it
from existing raw files. A numerical disagreement exits nonzero; performance
threshold crossings are reported as review flags, not silently accepted.

The machine was an AMD Ryzen 9 9950X, Linux 6.17.0-40, CPython 3.13.12,
NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2 and Matplotlib 3.10.8.
OpenMP, OpenBLAS and MKL thread counts were all one. Timing workers were pinned
to logical CPU 31. This was a shared workstation, not a reserved machine;
worker load averages are retained. Affinity restricts these workers and does
not reserve that CPU from other processes.

Seven paired rounds alternate baseline/current execution order. Each worker
warms each case, then runs fresh analyses until at least three executions and
0.10 seconds of measured work have accumulated. Each reported observation is
the median within that worker; the table gives the median and interquartile
range of those seven observations. Transformation preparation has 25 fresh
measurements per worker. Imports, model construction, garbage collection and
profiling are outside the reported workload timings.

**Analysis construction and `run()` are timed together.** PySTRA 1.6.0 executes
FORM inside the SORM and importance-sampling constructors; omitting construction
would favor the baseline unfairly. Each execution uses a new model and analysis.
A small callback counter independently counts physical limit-state evaluations,
including internal FORM runs. Transformation preparation times `init_run()` on
a fresh FORM object: correlation calibration and factorization, plus the
version's runtime initialization checks.

Both simulations use 20,000 samples, blocks of 1,000, unit proposal scale and
zero target CoV to exhaust the sample budget. The baseline global NumPy stream
and the current local Generator both use seed 20260913, but their streams differ.
The same stream is recreated for each repeated workload in its own version.

## Fixed cases

| Case | Model and failure event |
| --- | --- |
| `form_normal` | Independent Normal R(10, 2), S(5, 1); R-S <= 0 |
| `form_nonnormal` | Independent Lognormal R(10, 2), Gumbel S(5, 1); R-S <= 0 |
| `form_correlated` | The same Lognormal/Gumbel model, physical Pearson correlation 0.4 |
| `sorm_nonlinear` | Lognormal X1(500, 100), Normal X2(2000, 400), Uniform X3(5, 0.5); 1-X2/(1000 X3)-(X1/(200 X3))^2 <= 0; correlations 0.3, 0.2, 0.2; curve-fit SORM |
| `crude_mc` | The correlated Lognormal/Gumbel model, crude Monte Carlo |
| `importance_sampling` | The correlated Lognormal/Gumbel model, proposal centered at the computed FORM point |

Parentheses contain mean and standard deviation. Input units are consistent
within each limit-state expression. Default FORM convergence and differentiation
settings are retained in both versions. These inexpensive callbacks expose
framework overhead; the relative effect can be smaller when an external solver
dominates evaluation cost.

## Runtime before and after optimization

Times are milliseconds, as median [first quartile, third quartile]. Each
percentage compares with the baseline measured in the same paired run. The
initial artifact retains the original baseline timings; the table below shows
the newly measured baseline alongside both sets of current timings. This shared
workstation has scheduling and clock variation between measurement sessions.

| Case | 1.6.0, new run | Before optimization | After optimization | Before vs 1.6.0 | After vs 1.6.0 | Evaluations, both |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal FORM | 0.715 [0.663, 0.722] | 0.878 [0.856, 0.899] | 0.632 [0.624, 0.647] | +30.4% | -11.6% | 12 |
| Non-normal FORM | 1.981 [1.911, 1.994] | 2.679 [2.592, 2.721] | 1.252 [1.227, 1.266] | +39.4% | -36.8% | 39 |
| Correlated FORM | 2.976 [2.923, 3.008] | 3.775 [3.739, 3.836] | 2.191 [2.128, 2.213] | +30.8% | -26.4% | 57 |
| Nonlinear SORM | 6.581 [6.490, 6.662] | 7.469 [7.418, 7.711] | 5.449 [5.340, 5.462] | +16.4% | -17.2% | 180 |
| Crude Monte Carlo | 491.529 [491.120, 493.270] | 527.179 [523.549, 534.464] | 74.283 [74.154, 74.804] | +7.9% | -84.9% | 20,000 |
| Importance sampling | 501.792 [497.189, 502.588] | 641.582 [634.971, 653.505] | 78.103 [76.543, 78.314] | +29.3% | -84.4% | 20,057 |

The normal FORM index remains 2.2360679774990433. The non-normal and correlated
FORM indices agree with the pre-optimization values to floating-point round-off.
The SORM difference from 1.6.0 remains within the existing `2e-4` beta tolerance
for the curvature kernel; optimization does not change that comparison policy.
Independent simulation streams retain the five-pooled-standard-error criterion.

| Simulation | 1.6.0 samples/s | Before samples/s | After samples/s | After throughput change vs 1.6.0 |
| --- | ---: | ---: | ---: | ---: |
| Crude Monte Carlo | 40,689 | 37,938 | 269,240 | +561.7% |
| Importance sampling | 39,857 | 31,173 | 256,073 | +542.5% |

These rates include preparation and, for importance sampling, its FORM run.
The 20,000-sample retained-storage workflow and evaluation counts are unchanged.

## Transformation preparation

| Case | 1.6.0, ms | Before, ms | After, ms | After vs 1.6.0 |
| --- | ---: | ---: | ---: | ---: |
| Normal FORM | 0.375 | 0.365 | 0.123 | -67.1% |
| Non-normal FORM | 0.515 | 0.728 | 0.134 | -73.9% |
| Correlated FORM | 0.982 | 1.176 | 0.890 | -9.4% |
| Nonlinear SORM | 1.909 | 1.888 | 1.918 | +0.5% |
| Crude Monte Carlo | 0.959 | 1.179 | 0.909 | -5.2% |
| Importance sampling | 0.966 | 1.203 | 0.885 | -8.4% |

The three correlated two-variable cases use the same preparation model.
Zero-correlation pairs now skip quadrature whose computed values were unused.
SORM's preparation measurement still times a fresh FORM initialization; its
complete runtime also benefits from reusing that preparation within SORM.

## Stored samples and peak allocations

Memory is measured separately with `tracemalloc`, after imports/model creation
and garbage collection, over analysis construction and execution. Each case has
three measurements; the reported peak is their median. NumPy data allocations
are visible to this tracer in this environment. This is the incremental traced
allocation peak, not total resident memory or every native solver allocation.
Raw repetitions remain in the generated evidence.

Each simulation retains **960,000 bytes in both versions** for `u`, `x`,
limit-state values and radial distances: `(2 * dimension + 2) * samples * 8`.
Convergence histories still occupy 320,000 bytes in 1.6.0 and 480,000 bytes in
2.0, a **50% increase**. The additional `_log_q_bar` is 160,000 bytes and
preserves tail probabilities when ordinary probabilities underflow.

| Case | 1.6.0 peak, bytes | Before peak, bytes | After peak, bytes | After vs 1.6.0 |
| --- | ---: | ---: | ---: | ---: |
| Normal FORM | 47,024 | 46,992 | 15,168 | -67.74% |
| Non-normal FORM | 48,354 | 50,550 | 23,530 | -51.34% |
| Correlated FORM | 84,642 | 86,726 | 83,824 | -0.97% |
| Nonlinear SORM | 109,625 | 118,201 | 99,580 | -9.16% |
| Crude Monte Carlo | 1,662,888 | 1,830,335 | 1,823,069 | +9.63% |
| Importance sampling | 1,667,525 | 1,841,807 | 1,833,534 | +9.96% |

Neither optimized simulation peak exceeds the 10% threshold in this run.
The two remaining flags are the 50% history increases. Their allocation is
explicit in the arrays' recorded byte counts; result diagnostics and the
retained FORM record contribute smaller allocations.
Reducing histories to block-end entries would require a separate storage change
and is not part of this runtime optimization. Repeated sample-array append
behavior is also unchanged, so these measurements are not a scaling guarantee.

## Changes and profile evidence

The [original profiles](performance-profiles/before/) and
[optimized profiles](performance-profiles/) accompany the timing evidence.
Profiles use 25 executions for each FORM/SORM case and one for each simulation;
their instrumented elapsed times must not replace the timing table.

* Generic scalar marginal transformations bypass array partitioning in the
  central region. Array transformations no longer calculate placeholder PPF
  values for points that will instead use a tail inverse. Survival probabilities,
  log-tail inversion and inverse checks remain in the generic tail path.
* Native Gumbel quantiles use the same closed-form inverse with logarithms of
  normal tail probabilities in extreme tails. This removes per-point SciPy distribution
  dispatch, which dominated the Gumbel sampling profiles. Extension subclasses
  retain the generic path so their quantile overrides still apply. The Uniform
  scalar map also avoids array selection overhead.
* Independent correlation pairs no longer create unused quadrature grids.
  Correlated pairs retain the original quadrature and optimization procedure.
* Finite differences retain the same point order, step sizes, physical
  evaluations and callback input isolation, using slices and grouped arrays
  instead of repeated index lists. Shape, signature and finiteness validation
  remain. Error context is formatted only when an error is reported.
* SORM uses the transformation prepared by its own fresh FORM run. This is
  reuse within one run; a later run recomputes it. Supplied-FORM coordinate
  checks, custom initialization overrides and the normal-space requirement
  remain active. No reuse cache spans changing input models or options.

There are no remaining runtime regressions above 10% in these fixed cases.
Tail checks include scalar/batch agreement at the branch boundaries for 15
families, mpmath Gumbel references, logarithmic quantiles beyond double-probability
underflow, generic inverse refinement and the high-reliability sampling cases.
The final full-suite count and installed-example results are recorded with the
[release checks](release-checks.md). Repeat these measurements on the final
integrated release commit and supported platforms.
