# Performance against PySTRA 1.6.0

Measured September 13, 2026, for the C6e release review. The benchmark began at
`7f41c6c` on `v2.0-contracts`; subsequent profiling overlapped Claude's documentation
commit `c0fd369`. Both have the same library source as `bcd77e9`. The reference
is the exact `v1.6.0` export used for the [migration trials](trials.md).
Source hashes and import paths are captured in the
[generated results](performance-results.json).

**Outcome:** evaluation counts are unchanged in all six cases and numerical
comparisons pass. Five workloads have runtime regressions above the provisional
10% threshold. Four non-normal transformation-preparation measurements also
exceed it. Stored sample arrays have unchanged size; importance sampling's
traced peak allocation rises 10.5%, principally because of an additional
log-probability history. These findings require a maintainer decision or a
separately validated optimization before closing the performance gate.
No library code was changed for this package.

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
of that summary, without hand-edited measurements. `--summarize` recomputes it
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

## Runtime and evaluation counts

Times are milliseconds, as median [first quartile, third quartile]. The change
is current/baseline minus one. Bold changes exceed 10%.

| Case | 1.6.0 time | Current time | Change | Evaluations, both versions |
| --- | ---: | ---: | ---: | ---: |
| Normal FORM | 0.673 [0.663, 0.679] | 0.878 [0.856, 0.899] | **+30.4%** | 12 |
| Non-normal FORM | 1.921 [1.892, 1.948] | 2.679 [2.592, 2.721] | **+39.4%** | 39 |
| Correlated FORM | 2.886 [2.848, 2.903] | 3.775 [3.739, 3.836] | **+30.8%** | 57 |
| Nonlinear SORM | 6.415 [6.349, 6.552] | 7.469 [7.418, 7.711] | **+16.4%** | 180 |
| Crude Monte Carlo | 488.457 [484.737, 498.987] | 527.179 [523.549, 534.464] | +7.9% | 20,000 |
| Importance sampling | 496.165 [492.637, 506.261] | 641.582 [634.971, 653.505] | **+29.3%** | 20,057 |

The normal FORM index is exactly matched at 2.2360679774990433; the other FORM
indices match at 2.4115594883556053 and 3.11290884065735. SORM differs by
0.000135216 in beta, within the 0.0002 tolerance already declared for the
migration trial's curvature calculation. The Monte Carlo probability
comparisons are within 0.79 pooled reported standard errors; the criterion is
five. These checks protect against comparing a faster but different computation.

| Simulation | 1.6.0 samples/s | Current samples/s | Throughput change |
| --- | ---: | ---: | ---: |
| Crude Monte Carlo | 40,945 | 37,938 | -7.3% |
| Importance sampling | 40,309 | 31,173 | **-22.7%** |

These rates include preparation and, for importance sampling, its FORM run.
They measure the complete retained-sample workflow, not just random-number
production. The sample count is fixed in both versions.

## Transformation preparation

| Case | 1.6.0, ms | Current, ms | Change |
| --- | ---: | ---: | ---: |
| Normal FORM | 0.359 | 0.365 | +1.7% |
| Non-normal FORM | 0.493 | 0.728 | **+47.7%** |
| Correlated FORM | 0.973 | 1.176 | **+20.9%** |
| Nonlinear SORM | 1.891 | 1.888 | -0.2% |
| Crude Monte Carlo model | 0.942 | 1.179 | **+25.1%** |
| Importance-sampling model | 0.940 | 1.203 | **+27.9%** |

The last three two-variable correlated cases use the same model. Their
preparation measurements vary slightly with scheduling, but all show the
same regression. The non-normal independent case also prepares a quadrature
grid in the existing implementation.

## Stored samples and peak allocations

Memory is measured separately with `tracemalloc`, after imports/model creation
and garbage collection, over analysis construction and execution. Each case has
three measurements; the reported peak is their median. NumPy data allocations
are visible to this tracer in the measured environment. This is the incremental
traced allocation peak, not total resident process memory or all possible native
solver allocations. Raw repetitions are included in the generated results.

For each simulation, retained `u`, `x`, limit-state and radial-distance arrays
total **960,000 bytes in both versions**: `(2 * dimension + 2) * samples * 8`.
Convergence histories occupy 320,000 bytes in 1.6.0 and 480,000 bytes currently,
a **50% increase**. The added `_log_q_bar` array alone is 160,000 bytes.

| Case | 1.6.0 traced peak, bytes | Current traced peak, bytes | Change |
| --- | ---: | ---: | ---: |
| Normal FORM | 47,024 | 46,992 | -0.1% |
| Non-normal FORM | 48,354 | 50,550 | +4.5% |
| Correlated FORM | 84,598 | 86,726 | +2.5% |
| Nonlinear SORM | 109,669 | 118,201 | +7.8% |
| Crude Monte Carlo | 1,664,632 | 1,830,335 | +9.95% |
| Importance sampling | 1,667,328 | 1,841,807 | **+10.5%** |

Crude Monte Carlo is close to the threshold and should be monitored. The
importance-sampling increase is 174,479 bytes, of which 160,000 are explained
by the added log history; the remainder includes result diagnostics, retained
FORM state and temporary allocations. Exact peaks vary by a few kilobytes.
Both versions append retained samples repeatedly, so this evidence is limited
to the stated sample/block sizes and is not a memory-scaling guarantee.

## Causes and candidate fixes

The retained [profiles](performance-profiles/) support the following attribution.
Profiles are separate instrumented runs: 25 executions for each FORM/SORM case
and one for each simulation. Their wall times include profiler overhead and
must not replace the uninstrumented timing table.

1. **Normal FORM runtime (+30.4%, about 0.205 ms):** the normal marginal maps
   and evaluation count are unchanged. The current path adds evaluator shape,
   finiteness and signature handling, a checked result record and a model-state
   snapshot for safe reuse. The profile has 34,975 calls over 25 analyses versus
   21,875 in 1.6.0; the evaluator and `_problem_state` account for identifiable
   new work. Candidate: prepare immutable callback metadata once per run and
   avoid redundant validation inside already validated numerical kernels.
   Preserve public checks and failure diagnostics.
2. **Non-normal FORM runtime (+39.4%) and correlated FORM (+30.8%), plus all
   four flagged preparation measurements:** `zi_and_xi` transforms quadrature
   nodes spanning [-6, 6]. The current generic Gumbel mapping partitions tails,
   calls checked quantile adapters and uses survival quantiles; its Jacobian
   can use logarithmic densities. The baseline directly called a SciPy PPF.
   The profiles show that added distribution work alongside evaluator checks;
   preparation regressions are concentrated in the Gumbel cases. Candidates:
   provide a verified closed-form Gumbel normal-to-physical map, avoid computing
   placeholder PPF values for entries that will use a tail quantile, and reuse
   immutable quadrature nodes/weights. Retain accuracy in extreme tails.
3. **SORM runtime (+16.4%, about 1.054 ms):** both versions use 180 evaluations,
   21 transformation Jacobians and two correlation preparations. Preparation
   itself is unchanged within noise. Extra evaluator/Jacobian checks and FORM
   state/result handling appear in the current profile, including reuse checks
   and snapshot construction. Candidate: share validated evaluation and prepared
   transformation state between the underlying FORM and SORM run, with explicit
   invalidation when model/options change. Do not remove failure or reuse checks.
4. **Importance-sampling runtime (+29.3%; throughput -22.7%):** moving the
   proposal to the design point puts many more Gumbel coordinates above the
   `u > 3` tail switch than crude Monte Carlo. The profile records 3,971 generic
   ISF calls in the current IS workload, versus no such calls in 1.6.0; crude
   Monte Carlo has only 23. The current mapping first evaluates a placeholder
   PPF and then the ISF for those tail points. Per-point array conversion and
   partitioning inside the 20,000-iteration transformation loop amplify the
   cost. Candidates: apply the verified marginal transformation to a whole
   block, or add a safe scalar fast path, while retaining survival/log-tail
   handling. This is the dominant simulation regression; the new vectorized
   log-weight reduction is not the dominant cost.
5. **History storage (+50% for both simulations) and IS peak (+10.5%):**
   `_log_q_bar` preserves information when probabilities underflow, but it is
   allocated for every sample although only block-end entries are used for
   history. Candidate: store all three histories at block ends, with a
   consistent diagnostic schema; separately consider preallocated sample
   storage or an explicit optional-storage policy. These require their own
   correctness and memory-scaling checks.

These are source/profile-based explanations, not measured speedup claims for
unimplemented fixes. The attribution does not assign every elapsed microsecond
to a single cause. Repeat the benchmark after any optimization and on the final
release commit; retain the numerical and tail-accuracy checks alongside it.
