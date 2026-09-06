# Calibration review for PySTRA 2.0

Reviewed 6 September 2026 at `13cc782`. This records current behavior and the
recommended migration direction; the runtime implementation is unchanged.

## Primary workflow

The maintainer identifies the normalized code-design study implemented by
`GenericCalibration` as the principal code-calibration workflow. Make that
workflow the starting point for the public API:

1. Specify a probability model and characteristic values.
2. Specify candidate code factors and a design rule.
3. Design across representative load ratios or cases.
4. Evaluate the reliability of those designs and compare against targets.
5. Compare or adjust the candidate factors, then verify the selected designs.

The existing implementation already separates `design()` from
`analyse_design()`. Its normalized G/P/Q formulation and tutorial should be
the first migration example. Currently it evaluates supplied factors; it does
not automatically fit factors to a target. `beta_t` is only a plotting input.
Automated fitting would need an explicit objective, weights, bounds, and
acceptance criteria; none should be silently chosen during extraction.

The older `Calibration` is a specialized inverse workflow: solve each case's
design parameter to a target reliability, then derive partial and combination
factors from the design points. Preserve those numerical methods and reference
examples as separately usable tools. They must not impose their data model or
solve/derive/select sequence on normalized code-design studies.

## Confirmed findings

The existing calibration/FBC/load-combination tests pass: 18 tests, with nine
legacy-constructor warnings. They do not cover the following reproductions.
No tests under `tests/` directly exercise `GenericCalibration` or
`GenericModel`; the executed generic-calibration tutorial covers the normal
successful workflow.

| Priority | Finding | Evidence and consequence |
| --- | --- | --- |
| High | Case ordering changes factor assignments | Starting with `tests/test_calibration.py::setup1`, reverse only the insertion order of `dict_comb_cases`, leaving the action order unchanged. Run matrix calibration and realign output rows by case name. The Q1-leading case changes from Q1/Q2 factors `1.0/0.898165` to `0.898165/1.0`; the maximum factor-table change is `0.101835`. `calc_epg_s_mat()`, `_get_psi_row_mat()`, and `get_psi_max()` infer leading actions from table diagonals rather than case metadata. |
| High | Preferred explicit cases lose calibration roles | Recreate the same fixture with `LoadCombination(lsf=lc.lsf, cases=lc.cases, constants=lc.constant)`. Resistance and other-action labels are empty; R and G are classified as combination variables. Coefficient calibration returns a `(2, 0)` resistance-factor table, then `get_design_param_factor()` raises `IndexError`. `_init_from_cases()` invents legacy metadata that cannot express the actual roles. The Turkstra factory restores roles separately, so behavior depends on construction path. |
| High | Failed FORM is accepted by generic calibration | Limit the real FORM instance constructed by `analyse_design()` to one iteration in a diagnostic harness. Both `converged` and `results_valid` are false, but `get_reliabilities()` stores a finite beta (approximately `-0.00868` in the check). FORM emits a warning; the returned grid contains no failure status and `analyse()` can mark the study analyzed. The old target solver also reads FORM results without checking validity. |
| High | Generic study results become stale when factors change | Analyze a one-point grid (`aq=ag=0.5`), then change the registered model's phi from `0.8` to `1.0`. The stored beta remains `4.12343` and `is_analysed` remains true, while fresh evaluation gives `2.61487`. The model is retained by reference and setters do not invalidate study results. The same issue applies to changed grids/model inputs. |
| Medium | Design parameter name is not honored throughout | Rename the fixture's limit-state parameter and constant from `z` to `k`, and set `calib_var='k'`. Matrix calibration fails with `KeyError: ['k'] not in index`. `_run_calibration()` always stores a `z` column; `calc_lsf_eval_df()` also supplies literal `z`. |
| Medium | Outer solve failure has no result contract | Call `_calibration_optimize()` on fixture 1 with `max_iter=1`. It returns a design near `2.02223`, with beta `2.49061` against target `4.3`, after an fsolve warning. No solve status or residual accompanies the returned pair. The alpha path similarly prints and returns when its iteration limit is reached. |

The generic-model checks used unit means, Lognormal model errors with standard
deviations 0.05 and 0.10, Lognormal resistance with standard deviation 0.08,
Normal G/P/Q with standard deviations 0.08/0.10/0.10, unit characteristic
values, and factors phi/gamma_g/gamma_p/gamma_q = 0.8/1.2/1.5/1.6. The failed
FORM check followed the change to phi=1.0. These values reproduce software
behavior; they are not recommended engineering inputs.

## Quick naming audit

An identifier-token scan of all 45 runtime Python modules, excluding comments
and strings, found the following distinct spellings and code occurrences.
These are review candidates, not an automatic rename list; occurrences count
definitions and uses, not independent variables or required edits.

| Pattern | Distinct spellings | Code occurrences |
| --- | ---: | ---: |
| `dict_` | 8 | 55 |
| `list_` | 9 | 37 |
| `arr_` / `array_` | 3 | 9 |
| Prefixed/compound `df` names, including `dfXstar` and `dfpsi` | 29 | 189 |
| Total | 49 | 290 |

All these runtime matches are in `calibration.py` and `loadcomb.py`; 231 of
the 290 occurrences belong to the old `Calibration` class. None of these
prefix patterns occurs in `GenericModel` or `GenericCalibration`, though the
latter has the separate `model_dict` record discussed above. The nine `list_`
spellings here are container labels (`list_form_obj`, `list_z_cal`,
`list_dist_resist`, etc.), rather than list-producing operation names.

The same patterns occur 163 times across three test files and 80 times in code
cells across three tracked notebooks (including legacy notebooks). Tracked
Python examples and maintenance scripts have no matches. Documentation prose
and notebook outputs were excluded; updating them adds follow-through work.

The scan deliberately excludes bare `df`, NumPy's `array_equal`, and conversion
names such as `to_dict` or `as_dict`. In copula/distribution code, `df` also
means degrees of freedom and must not be mistaken for a DataFrame name.
Readable local DataFrame names need individual judgment. Domain concepts and
operation names containing words such as list or set are not type-prefix
violations. This is a concentrated cleanup that should accompany replacement
of the old workflow, not a repository-wide word-substitution exercise.

## Dictionary and case design

The problem is hidden schemas and duplicated meaning, rather than dictionaries
as a Python facility:

- The constructor's `dict_dist_comb` maps action names to `max`/`pit`
  distributions, while the stored attribute of the same name maps case names
  to complete variable sets.
- `dict_comb_cases` means leading-action names in the legacy/Turkstra paths,
  but all variable names in the explicit-case path.
- `dict_label`, `label_*`, `distributions_*`, `cases`, and `dict_dist_comb`
  duplicate related state, while factor matrices additionally rely on order.
- `eval_lsf_kwargs()` fills unspecified stochastic variables with zero to
  isolate resistance/action contributions. That is an algorithmic assumption,
  not an appropriate general contract for evaluating an arbitrary limit state.
- `stochastic_model(**kwargs)` silently ignores unknown override names, so a
  misspelled design parameter can leave the design unchanged.

Keep mappings where names are the natural keys, such as nominal values or
named cases. Use small validated records for concepts with a fixed schema:
code factors, normalized model inputs, a case's leading actions, and results.
Use semantic names such as `nominal_values`, `cases`, and `leading_actions`.
Do not encode container types in names, including local variables and private
helpers. Remove pseudo-Hungarian prefixes (`dict_`, `list_`, `arr_`)
and expand cryptic remainders into meaningful names. Express types through
annotations. Conventional `df` or a readable `factors_df` is acceptable in
short local pandas operations; this does not justify cryptic names such as
`dfXstarcal` or type-driven public interfaces. Do not rebuild the same schema
in parallel label lists.
Reject missing/unknown names and preserve ordering only
for array alignment, never to infer engineering roles.

`LoadCombination` currently combines probabilistic case generation, variable
storage, correlation setup, FORM execution, and special-purpose limit-state
evaluation. Preserve inspectable probabilistic cases and FBC/Turkstra case
generation. Reduce the object to those responsibilities, or replace the
generator with a function returning explicit cases if that makes the examples
simpler. The exact public name is still open. Keep a code's factored design
combination distinct from a probabilistic leading/companion-action case.
Neither is a mandatory container for the normalized generic workflow.

## Implementation direction

Start with the existing generic tutorial: same normalized equations and
numerical outputs, but validated model/factor inputs and an explicit returned
study result. Separate the probability model from candidate code factors so
comparing factor sets does not require duplicating or mutating a model. Replace
`GenericModel.get()`'s 14-position tuple with named access. Move plotting color,
labels, and range styling out of numerical model registration.

Return design values and reliability results at each grid point, including
convergence, target margins when a target is supplied, and explicit coordinates.
Result snapshots must remain associated with the exact analyzed inputs.
Permit solver settings and explicit dependence assumptions; the current
generic runner always constructs a default FORM with independent marginals.
Validate complete models, finite factors/nominals, admissible load ratios, and
nonzero design denominators before starting a grid.

Extract the old target solve and coefficient/matrix factor derivation as
specialist operations. Declare the design-rule assumptions behind term
isolation and factor extraction; the currently tested nonlinear examples use
multiplicative model errors and do not establish support for arbitrary coupled
nonlinear limit states. Fix case-role alignment and failure handling during
that extraction, preserving the published-reference regression examples.

Before runtime replacement, add meaningful regression cases for the confirmed
failures above, analytic normalized-design/reliability cases, grid endpoints,
invalid inputs, and comparisons of factor sets on a shared probability model.
Do not discard failed cases when reporting a reliability envelope. Exact class
names and any automated factor-fitting policy remain design decisions.
