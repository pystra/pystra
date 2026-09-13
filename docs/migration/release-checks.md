# Installed-package release checks

Recorded September 13, 2026, for `v2.0-contracts` at `c0fd369` (internal
version `2.0.0a1`). This is evidence for the C6e review, not approval to publish.
No library or example code was changed for these checks.

**Outcome:** wheel/sdist construction, isolated installation, public imports,
converter CLI and the analytic FORM/SORM/Monte Carlo session pass. Five of six
core example scripts pass unchanged. The threaded example fails on obsolete
array indexing, so the complete installed-examples gate remains **failed**.
The optional OpenSees check is **blocked** by unavailable offline dependencies.

## Repeat the checks

From the checkout, using a Python interpreter compatible with the package:

```sh
python scripts/check_installed_package.py \
  --output /tmp/pystra-installed-checks --with-opensees
```

The output directory must be new and outside the checkout. The script uses
`uv build` to build an sdist and then its wheel in an isolated build environment,
creates a fresh virtual environment, installs the wheel and its dependencies,
and copies the maintained examples byte-for-byte into an external run directory.
Every Python check runs there with `-I`, without `PYTHONPATH`, `PYTHONHOME`,
user site-packages or access to the checkout through an editable installation.
The virtual environment uses its base interpreter's standard library, as usual.

The script retains exact commands, logs, artifact hashes, dependency versions,
example hashes and outcomes in `results.json`. It exits nonzero for a core
failure; optional OpenSees failures are recorded separately. The checked-in
[results](release-checks-results.json) are a copy of that generated file.
Paths, timings and archive hashes may change in a later run; archive timestamps
are not normalized for byte-identical builds.

A wheelhouse can support an offline repeat:

```sh
python scripts/check_installed_package.py \
  --wheelhouse /path/to/dependency-wheels \
  --output /tmp/pystra-installed-offline --with-opensees
```

### Environment used here

The initial package-index attempt failed with a DNS lookup error for
`pypi.org`. The successful build and installation used compatible distributions
already present in uv's PyPI wheel cache:

```sh
taskset -c 0 python scripts/check_installed_package.py \
  --uv-cache-source /home/ccaprani/.cache/uv \
  --output /tmp/pystra-c6e-installed --with-opensees
```

`--uv-cache-source` selects the newest locally cached compatible wheel for
each dependency, follows its active dependency declarations, verifies every
hashed `RECORD` entry and repacks those files into a private wheelhouse. The
normal uv resolver then checks their requirements. This uses cached wheel
contents; it does not copy the host's installed packages. The original cache
is read-only, and the new environment has no system site-packages. The saved
JSON records the cache archive, version, repacked hash and checked entry count
for every dependency. Cache layout support is specific to uv's `wheels-v6`.

The interpreter was CPython 3.13.12 on Linux x86-64, with uv 0.11.7.
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` were all `1`;
Matplotlib used `Agg`. The fresh environment resolved NumPy 2.5.3,
SciPy 1.18.1, pandas 3.0.5 and Matplotlib 3.11.1, plus their dependencies.
These are the available cached versions, not a minimum-dependency matrix.
The performance comparison uses a separate, shared 1.x/2.x dependency set.

The imported package was
`/tmp/pystra-c6e-installed/venv/lib/python3.13/site-packages/pystra/__init__.py`.
The smoke check confirms an active virtual environment, disabled user site,
and a package path under that environment. It imports all 70 top-level exports
and nine public modules, including calibration, decision, active learning and
migration. Active-learning algorithms are not exercised without their optional
scikit-learn dependency.

## Artifacts and metadata

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `pystra-2.0.0a1-py3-none-any.whl` | 232,730 bytes | `191063f8faad382812dcb0845f2bfa2c515e77fc990b1ec102241053756e540b` |
| `pystra-2.0.0a1.tar.gz` | 472,070 bytes | `22093dd439f177158dd451dd28999ca5879f685213577ad010dcb982ddd3cc03` |

Both artifacts include the converter's generated data. The sdist excludes
`.claude` and `.codex-commits` scratch/coordination files. The wheel archive's
CRC check and `uv pip check` pass. Metadata declares Python `>=3.12`, the four
runtime dependencies, README type `text/x-rst`, and both `LICENSE` and
`THIRD_PARTY_NOTICES`. The package retains its internal version label; release
versioning remains a separate maintainer task.

An additional read-only metadata check passed for both artifacts:

```sh
python -m twine check /tmp/pystra-c6e-installed/dist/*
```

Twine is a host-side release tool and is not installed into the tested runtime
environment. This additional command is separate from the repeat script.

## Executed checks

| Check | Outcome |
| --- | --- |
| Isolated build: sdist, then wheel from sdist | Pass |
| Fresh venv wheel installation and dependency consistency | Pass |
| Public imports and package provenance | Pass |
| `python -I -m pystra.migrate --help` | Pass |
| Installed `convert_source` import/class/keyword rewrite | Pass |
| Short independent-normal FORM/SORM/MC session | Pass |
| `examples/ddm_example.py` | Pass |
| `examples/example.py` | Pass, including 100,000-sample distribution analysis |
| `examples/gev_example.py` | Pass |
| `examples/sensitivity.py` | Pass |
| `examples/timing.py` | Pass, retaining its full 100 repetitions per variant |
| `examples/example_parallel_multithreading.py` | Fail: invalid two-dimensional indexing |
| Optional `openseespy` installation and `examples/openseespy_ex.py` | Blocked: dependency absent from offline wheelhouse |

The short session uses independent normal resistance `(mean=10, std=2)` and
load `(mean=5, std=1)`, with failure `R-S <= 0`. Its exact index is `sqrt(5)`
and probability is `0.012673659338734126`. FORM returns
`beta=2.2360679774990433`; SORM returns `0.012673659338758572`.
Crude Monte Carlo with 20,000 samples, block size 1,000, `target_cov=0` and
seed 20260913 returns `0.01305`, about 0.48 analytic standard errors from the
reference. Deterministic assertions use relative tolerances `1e-9` for beta
and `1e-8` for SORM probability; the stochastic check allows five analytic
standard errors.

## Release findings

1. **Threaded example:** the callback evaluates `len(X1[0, :])` and later
   `X1[0, i_thread]`, while each named variable now arrives as a one-dimensional
   sample vector. It raises `IndexError`, wrapped in `AnalysisError`, before
   returning a reliability result. Update it to `len(X1)` and indexed vector
   elements for all three variables, then rerun the unchanged installed package.
   A thread-only `queue.Queue` would also avoid the example's unnecessary
   multiprocessing queue. These are candidate example changes, not applied here.
2. **OpenSees:** dependency installation was attempted and the script was run;
   neither is counted as a pass. The network lookup is unavailable and the
   compatible dependency is not cached. Repeat in the explicit external-solver
   environment. The earlier [migration trials](trials.md) also record an MPI
   socket restriction when OpenSees is present in this sandbox.
3. **Final integration:** repeat on the final integrated release commit after
   the threaded example and queued library review fixes. This Linux/Python 3.13
   evidence does not replace the platform/dependency matrix, notebook execution,
   documentation checks or external-solver release job.
