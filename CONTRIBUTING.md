# Contributing to PySTRA

PySTRA implements traditional structural reliability methods and practical
tools for code calibration and structural assessment. Contributions should
make those methods easier to understand, verify, and use. General UQ features
need a specific structural reliability use case to fit the project.

This guide establishes the conventions for v3 onward. During the migration,
target `v3.0` for breaking changes and follow the
[migration plan](docs/v3.0-migration-plan.md). Existing code still contains
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
After 3.0, public API changes follow the documented deprecation and release policy.

## Naming and public interfaces

- Use snake_case for functions, methods, parameters, attributes, and compound
  module names; CapWords for classes; UPPER_SNAKE_CASE for constants.
- Follow the project acronym convention: `Form`, `Sorm`, `SystemForm`,
  `GevMax`, `FbcProcess`, `Ddo`, `Lqi`, `Swtp`. Use FORM/SORM/etc. in prose.
- Prefer `std`, `start_point`, `limit_state`, `model`, `options`, `n_samples`,
  `max_iterations`, `failure_probability`, and `reference_period`. Use `analyze`
  in Python identifiers. Keep established `pdf`, `cdf`, `ppf`, `beta`, and `alpha`.
- Name quantities by meaning. Avoid type-prefixed public names (`dict_nom`,
  `dfXstar`, `list_form_obj`) and compressed compound identifiers. Private
  helpers also follow the naming policy.
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

For calibration, follow the separate solve-designs, derive-factors,
select-factors, and verify-designs operations in the migration plan. Preserve
governing-case information and numerical diagnostics. A convenience runner
composes those operations; it does not duplicate them.

## Numerical contracts and evidence

Document coordinate systems, Jacobian direction, variable ordering, failure
event convention, and reference-period assumptions. Use the agreed 3.0 point
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
virtual environment using the dependency versions supported by that branch:

```sh
python -m pip install -e '.[test,docs]'
python -m pip install black
python -m pytest -q
```

Run Black on changed Python files and include formatting in the appropriate
commit. Black remains the formatter during migration. Naming/import lint checks
will be introduced as planned; this guide does not imply they are already
configured. Run the checks actually present on the target branch.

For documentation-only layout/link edits, build from `docs` using saved
notebook outputs:

```sh
python -m sphinx -b html -W --keep-going -D nbsphinx_execute=never source build/html
```

When changing notebook code or APIs used by a notebook, execute the affected
notebooks and inspect their outputs before building. Saved-output builds alone
do not validate tutorial execution. New methods should include a short runnable
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
