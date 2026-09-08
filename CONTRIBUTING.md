# Contributing to PySTRA

PySTRA implements traditional structural reliability methods and practical
tools for code calibration and structural assessment. Contributions should
make those methods easier to understand, verify, and use. General UQ features
need a specific structural reliability use case to fit the project.

This guide establishes the conventions for v2 onward. During the migration,
target `v2.0` for breaking changes and follow the
[migration plan](docs/v2.0-migration-plan.md). Existing code still contains
legacy APIs: their presence is not a style precedent. Implement the agreed
stage without introducing a second competing API. Changes intended for the
stable line retain its compatibility requirements.

## Start with the user and the numerical method

For a new method or substantial API change, include a short usage example,
the problem it solves, its assumptions, and a reference case. Put the proposed
contract in the PR before expanding the implementation. Routine fixes can
proceed directly with focused evidence; a separate design document is not
required for every change.

Keep numerical changes, mechanical renaming, formatting, and file moves in
separate commits wherever practical. Update the migration manifest/guide when
an import, parameter, result schema, default, or supported extension hook changes.
After 2.0, public API changes follow the documented deprecation and release policy.

## Naming and public interfaces

- Use snake_case for functions, methods, parameters, attributes, and compound
  module names; CapWords for classes; UPPER_SNAKE_CASE for constants.
- Follow the project acronym convention: `Form`, `Sorm`, `SystemForm`,
  `GevMax`, `FbcProcess`, `Ddo`, `Lqi`, `Swtp`. Use FORM/SORM/etc. in prose.
- Prefer `std`, `start_point`, `limit_state`, `model`, `options`, `n_samples`,
  `max_iterations`, `failure_probability`, and `reference_period`. Use `analyze`
  in Python identifiers. Keep established `pdf`, `cdf`, `ppf`, `beta`, and `alpha`.
- Name quantities by meaning. Do not use pseudo-Hungarian container/type
  prefixes such as `dict_`, `list_`, or `arr_`, or compressed forms such as
  `dfXstarcal`. This applies to parameters, attributes, private
  helpers, and local variables throughout the codebase, not only public APIs.
  Use `nominal_values`, `design_points`, or `reliability_results` rather than
  `dict_nom`, `dfXstar`, or `list_form_obj`. Replace cryptic abbreviations with
  meaningful names; deleting the prefix alone is not sufficient. Express
  container types through type annotations. Conventional `df` or a readable
  name such as `factors_df` is acceptable for a local DataFrame in a short
  pandas operation when it aids clarity. Prefer semantic names for parameters,
  persistent attributes, and returned results. Judge meaning rather than
  banning words: `list_cases()` describes an operation, and `cut_sets` describes
  a reliability concept. These are different from naming a variable
  `list_cases` merely because its container happens to be a Python list.
- Mathematical local names such as `x`, `u`, and `rho` are appropriate near an
  equation. Preserve user-provided variable names and external API spellings;
  a limit-state argument `R` can remain `R` when it names the model variable.
- Use properties for cheap data access and verbs for operations. Prefer
  explicit coordinates to boolean switches such as `getDesignPoint(False)`.
- Make configuration keyword-only, validate it, and reject unknown settings.
  Avoid booleans or numeric flags that silently change a parameter's meaning.
- Use explicit imports and deliberate `__all__` exports. Public signatures
  have type hints and NumPy-style docstrings covering shapes, units, defaults,
  assumptions, returns, and failure behavior. Mark internal helpers with `_`.

## Keep computation and state understandable

Models describe a problem; algorithms execute; results record what happened;
reporting presents those results. Constructors must not start an analysis.
Use explicit inputs and returned results rather than a sequence of methods
that populate attributes needed by later methods. Snapshot results so a later
run cannot change an earlier result.

Prefer small functions for numerical operations and small data objects for
specifications/results. Add a class when it owns a coherent responsibility.
Use composition and narrow interfaces where substitution is actually needed;
avoid deep inheritance trees, general registries, and speculative abstractions.

Keep arrays and explicit variable/case metadata in numerical kernels. DataFrame
conversion and plotting belong at the boundary. Do not use mutable tables as
hidden communication between solver stages. Return figures/axes without
implicitly displaying them, and return summaries rather than printing during
computation. Diagnostics can use logging or an explicit progress callback.

Do not mutate caller-owned models, mappings, arrays, or distributions while
iterating candidate designs. Keep caches, evaluation counts, and random state
local to the run, with documented invalidation and RNG ownership.

For code calibration, prioritize the normalized workflow: candidate factors,
code designs, reliability evaluation, and comparison against targets. Keep
probability models separate from factor sets. The specialized inverse workflow
uses the separate `solve_designs`, `derive_factors`, `select_factors`,
`design_with_factors` and `verify_designs` operations documented in the
[migration guide](docs/source/migrating.rst); do not force all code studies through it.
Preserve governing-case information and numerical diagnostics. A convenience
runner composes operations; it does not duplicate them.

## Numerical contracts and evidence

Document coordinate systems, Jacobian direction, variable ordering, failure
event convention, and reference-period assumptions. Use the agreed 2.0 point
and batch shapes at public boundaries; adapt existing kernels explicitly during
migration. Never infer array orientation from a square shape.

Distinguish estimates, approximations, bounds, and diagnostics. State what
event a bound applies to. Check the capabilities required by each algorithm;
unsupported inputs or transformations must fail clearly. Nonconvergence,
unattainable calibration targets, and sampling-budget termination must remain
visible to callers. They must not become a zero failure probability or a
previous run's successful result.

Use local random generators and explicit seeds for reproducible examples.
Specify the numerical environment and statistical tolerance when necessary.
Prefer published or analytic reference cases and mathematical invariants over
tests that merely repeat the implementation. For a bug, add a focused regression
case; for a method, test its assumptions and meaningful failure modes as well
as a successful example. Explain intentional changes to baseline values.

Preserve original references and attribution when extracting or reorganizing
existing methods. Review code against the shared project conventions and
numerical evidence, rather than a contributor's personal style.

## Local workflow and pull requests

Create a feature branch from the current integration branch. Install in a
virtual environment. Use Python 3.13 for the documentation/formatting tools;
the runtime test matrix currently retains Python 3.9 through 3.13:

```sh
python -m pip install -e '.[test,docs]'
python -m pip install black==26.5.1
python -m pytest -q
```

Run Black on changed Python files and include formatting in the appropriate
commit. Black remains the formatter during migration. Run
`python scripts/api_inventory.py --check-names --check-migration` for naming,
baseline API coverage, and current calibration signature checks. Update the
reviewed migration records alongside contract changes; historical naming maps
are not the current calibration API. Remaining attribute/parameter cleanup and
import enforcement
remain later stages; existing legacy fields are not a precedent for new code.
CI also executes every indexed tutorial using `scripts/execute_notebooks.py`.

Routine documentation builds use saved notebook outputs. From `docs`, run:

```sh
make html SPHINXOPTS="-W --keep-going"
```

When changing notebook code or APIs used by a notebook, execute the affected
notebooks and inspect their outputs before building. From the repository root,
for example:

```sh
python scripts/execute_notebooks.py ex_active_extensions --output-dir docs/source/notebooks
```

Omit the notebook name to execute all indexed tutorials, as CI does. To rerun
notebooks within Sphinx instead, use
`make html SPHINXOPTS="-E -W --keep-going -D nbsphinx_execute=always"` from `docs`.
Saved-output builds alone do not validate tutorial execution.
New methods should include a short runnable
example and theory/reference documentation. Keep optional external-solver or
network-dependent examples identifiable with their required environment.

PRs should state the concrete behavior change, API impact, numerical evidence,
and relevant validation commands/results. Identify skipped checks and known
limitations. Use the PR template as a review aid, applying only relevant items.
Avoid unrelated formatting or dependency churn. Maintainers handle release
version bumps; ordinary contributions should not bump the package version.

The migration plan defines the broader release matrix, benchmark comparisons,
packaging checks, and documentation gates. Small documentation edits do not
require rerunning the entire numerical suite.

## Licence and method provenance

PySTRA is GPL-3.0-or-later. Preserve copyright and licence notices when adapting
code, and identify its source and changes. Cite the original papers and benchmark
definitions; distinguish reproductions from variants and software comparisons.
Do not imply endorsement by the cited authors or toolbox developers. Include
independent reference calculations and explicit statistical tolerances for
stochastic methods. Run optional active-learning checks with `.[test,al]`.
