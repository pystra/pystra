Migrating from 1.x to 2.0
=========================

This page describes the API currently implemented on the ``v2.0`` development
branch (``2.0.0.dev0``). Naming, the calibration replacement, and the first
FORM result contract are implemented. Remaining algorithm results, options,
and parameter changes are described in the
:download:`migration plan <../v2.0-migration-plan.md>`.

Implemented naming changes
--------------------------

Functions and methods now use snake_case. Inconsistent class names follow the
project's CapWords convention. There are no aliases for the replaced names.
Most existing module imports retain their paths. Calibration is now a package
with explicit exports; its structural changes are described below.

.. list-table:: Representative changes
   :header-rows: 1
   :widths: 45 55

   * - 1.x
     - Current 2.0 development API
   * - ``model.addVariable(rv)``
     - ``model.add_variable(rv)``
   * - ``model.setCorrelation(matrix)``
     - ``model.set_correlation(matrix)``
   * - ``options.setImax(100)``
     - ``options.set_imax(100)``
   * - ``form.getBeta()``
     - ``form.get_beta()``
   * - ``form.getFailure()``
     - ``form.get_failure()``
   * - ``form.getDesignPoint(False)``
     - ``form.get_design_point(False)``
   * - ``form.showResults()``
     - ``form.show_results()``
   * - ``distribution.dF_dtheta(x)``
     - ``distribution.cdf_gradient(x)``
   * - ``joint.getTransformation(...)``
     - ``joint.make_transformation(...)``

The class names ``SystemFORM``, ``FBCProcess``, ``GEVmax``, ``GEVmin``,
``ScipyDist``, ``DDO``, ``DDOCriterion``, ``DDOObjective``, ``LQI``, ``SWTP``,
``SWTPRecord``, and ``SWTPIndexRecord`` are retained from v1.x. Acronyms in
CapWords retain their capitals, as recommended by
`PEP 8 <https://peps.python.org/pep-0008/#descriptive-naming-styles>`_.
Existing ``Form`` and ``Sorm`` also retain their names. Earlier v2 development
snapshots recased these names; those changes have been reversed.

The :download:`naming map <../migration/naming-map.json>` records the
reviewed spelling changes and retained class names. The :download:`definition manifest <../migration/api-migration.json>`
maps all 645 inventoried definitions, including private helpers. The
:download:`baseline inventory <../migration/api-baseline.json>` records
signatures, assigned attributes, dependency versions, and observed exports.
Inventory inclusion does not imply that an internal helper or an incidental
third-party export is a supported public API.

The distribution package now explicitly exports PySTRA classes. Import NumPy
and SciPy helpers directly from their own packages. Previously leaked SciPy
objects no longer shadow the ``beta``, ``gamma``, ``gumbel``, ``uniform``, and
``weibull`` distribution modules.

Current example
---------------

This example uses the implemented development API::

    import pystra as ra

    model = ra.StochasticModel()
    model.add_variable(ra.Normal("R", mean=10, stdv=1))
    model.add_variable(ra.Normal("S", mean=5, stdv=1))
    form = ra.Form(
        stochastic_model=model,
        limit_state=ra.LimitState(lambda R, S: R - S),
    )
    result = form.run()
    print(result.beta, result.converged)

The names ``stdv``, ``stochastic_model``, and the existing solver getters are
transitional. Scientific acronyms such as FORM and LQI retain their
conventional spelling in prose.

FORM result snapshots
---------------------

``Form.run()`` now returns a ``FormResult``. Its ``beta`` is the normal-equivalent
reliability index; ``geometric_beta`` retains the signed distance in
``standard_space``. These differ in Student-t standard space. A normal-space
index remains finite even when its very small probability underflows to zero.

``design_point`` (physical), ``standard_point``, and ``alpha`` are immutable
one-dimensional tuples in ``variable_names`` order. A later solver run cannot
change a previous result. Results include convergence, iteration count, and
limit-state/direction errors. Iteration exhaustion returns ``converged=False``
with unavailable probability, index, and point fields, and retains the existing
warning. Invalid input and numerical execution errors still raise. This does
not yet introduce a common failure policy for every algorithm.

Existing FORM getters remain available during migration. In particular,
``get_beta()`` still returns the geometric index; use ``result.beta`` when
comparing probability-equivalent targets across reference spaces. Other
algorithms have not yet migrated to this result contract.

Normalized code-calibration studies
-----------------------------------

``GenericCalibration`` is the primary code-calibration workflow. It evaluates
candidate factors against a probability model over a normalized G/P/Q load-ratio
grid. It does not silently choose a fitting objective or optimize the factors.

Replace model setters and the 14-position ``GenericModel.get()`` tuple with
explicit constructor fields. ``NominalValues`` holds characteristic values;
``CodeFactors`` holds a separate candidate factor set. Both validate positive,
finite values. ``GenericModel`` copies distributions/constants and optionally
accepts a copula. The copula follows the random-variable order
resistance_error, resistance, load_error, dead_load, permanent_load, live_load,
omitting constants. Independence is the default.

For example::

    from dataclasses import replace
    import pystra as ra

    model = ra.GenericModel(
        resistance=ra.Lognormal("R", 1, 0.08),
        dead_load=ra.Normal("G", 1, 0.08),
        permanent_load=ra.Normal("P", 1, 0.10),
        live_load=ra.Normal("Q", 1, 0.10),
        resistance_error=ra.Lognormal("w_R", 1, 0.05),
        load_error=ra.Lognormal("w_S", 1, 0.10),
        nominal_values=ra.NominalValues(1, 1, 1, 1),
    )
    factors = ra.CodeFactors(phi=0.8, gamma_g=1.2, gamma_p=1.5, gamma_q=1.6)
    study = ra.GenericCalibration(
        live_load_ratios=[0.2, 0.5, 0.8], dead_load_ratios=[0, 0.5, 1]
    )
    current = study.run(model, factors, target_beta=3.8)
    proposed = study.run(model, replace(factors, phi=0.9), target_beta=3.8)
    print(current.to_frame())
    fig, ax = ra.plot_calibration(
        {"Current": current, "Proposed": proposed}, target_beta=3.8
    )

This replaces ``add_model(...)``, ``analyse()`` and the study's mutable result
cache. Each ``run(model, factors, *, options=None, target_beta=None)`` returns a
new ``CodeCalibrationResult`` with the analyzed model snapshot, factors, designs,
and a ``FormResult`` at every grid point. ``beta`` is a fresh array of shape
``(n_dead_load_ratios, n_live_load_ratios)``. Failed points are retained as NaN;
their target margins are unavailable. ``to_frame()`` returns a reporting copy.
``plot_calibration`` accepts completed results and refuses to draw an envelope
that would hide failed points. Labels, colors and range annotations belong to
this plotting function, replacing styling in model registration.

Grids must be nonempty, one-dimensional, and within [0, 1], including endpoints.
FORM options can be supplied explicitly; the built-in normalized limit state
currently requires finite-difference derivatives and rejects DDM mode.

Explicit load cases
-------------------

``LoadCombination`` now stores inspectable probabilistic cases and metadata;
analysis is explicit through ``analyze_case(cases, case_name, options=...)``.
Its constructor accepts ``cases``, ``constants``, ``roles``, ``leading_actions``,
``limit_state`` and labelled Pearson ``correlation``. It validates inputs but
performs no reliability analysis. Case/model access returns independent copies.
Unknown variable overrides raise instead of being silently ignored.

Use ``VariableRoles(resistance=(...), other=(...), variable=(...))`` when case
roles matter. ``leading_actions`` maps each case name to its leading variable
action names. Explicit-case construction preserves these roles; case insertion
order never determines a leading action.

Replace the legacy nested ``dict_dist_comb``/``action_distributions`` constructor
with ``LoadCombination.from_actions(maxima=..., companions=..., resistance=...,
other=..., constants=..., limit_state=...)``. Maxima and companions are separate
mappings from action names to distributions. The default generates one
``<action>_max`` case per action. Use ``LoadCombination.turkstra`` for the retained
FBC reference-period/companion-duration construction.

The old ``lsf``, ``corr``, ``opt`` and positional constructor arguments are
removed: use ``limit_state``, ``correlation`` and per-evaluation ``options``.
``run_reliability_case`` is replaced by ``analyze_case``. Duplicate label groups,
mutable case-distribution attributes and special zero-filled limit-state
helpers are removed. Ordinary mappings remain appropriate for named cases,
nominal values and overrides; fixed records express roles and results.

Specialist design-point factor calibration
------------------------------------------

The old ``Calibration`` object is removed, without an alias. Its hidden sequence
of mutable DataFrame attributes is replaced by independently usable operations::

    problem = ra.FactorCalibrationProblem(
        cases, nominal_values=nominal_values, design_parameter="scale"
    )
    targets = ra.solve_designs(problem, target_beta=4.3, method="root")
    factors = ra.derive_factors(targets, method="matrix")
    selected = ra.select_factors(
        factors, resistance="minimum", loads="maximum", combinations="maximum"
    )
    designs = ra.design_with_factors(problem, selected)
    checks = ra.verify_designs(problem, max(designs.values), target_beta=4.3)

Here ``cases`` is a ``LoadCombination`` with explicit roles, leading actions,
a limit state and a common ``Constant("scale", ...)``. ``nominal_values`` maps
exactly the random-variable role names to positive characteristic values.
See the factor-calibration tutorial for complete linear and nonlinear examples.

``solve_designs`` supports ``method="root"`` (fsolve, or Brent with a supplied
``bracket``) and ``method="alpha"``. The default tolerance is 1e-4 in beta and
``max_evaluations=100`` limits FORM runs per case, including final verification.
Each ``CalibratedDesign`` carries the design value, reliability, target residual,
convergence, evaluation count and message. A failed inner or outer solve remains
in the result and cannot enter factor derivation. Solver success alone is
insufficient: the final beta residual must also satisfy the tolerance.

``derive_factors`` supports the original ``matrix`` and ``coeff`` methods.
These specialist methods require normal standard space and exactly one leading
case per variable action. Alpha projection also requires normal space. The
design rule assumes a separable resistance scale with multiplicative model
errors; designs are checked against the full limit-state residual. Arbitrary
coupled nonlinear functions are not supported by term isolation. The matrix
method retains the reference convention of evaluating each action coefficient
in its own leading case, then assembling by names. Singular systems raise.

``FactorSet`` contains immutable numeric rows with explicit case/variable axes.
``to_frame("resistance")``, ``to_frame("loads")`` and
``to_frame("combinations")`` return fresh tables. Selection defaults to
``"per_case"``; extrema must be requested explicitly, and governing-case ties
are retained. ``design_with_factors`` returns named design values. Selecting
their maximum is explicit in the example, followed by verification of every
case; extrema alone do not establish that targets are met.

Migration records
-----------------

The :download:`calibration naming map <../migration/calibration-naming-map.json>`
records the intermediate naming-only stage. It is historical: several of those
intermediate names have since been removed. The
:download:`calibration structure map <../migration/calibration-structure-map.json>`
records replacement operations and current definitions, signatures and result
fields. The definition manifest retains a disposition for every baseline
entry; removed/replaced APIs have no misleading one-to-one current name.
These maps are not global text-substitution recipes for user scripts.

The :download:`calibration review <../calibration-review.md>` records the original
reproductions and their resolution. Readable local ``df`` names, statistical
degrees of freedom and conversion methods such as ``to_dict()`` retain their
meanings; pseudo-Hungarian container prefixes are removed.

Integration and validation
--------------------------

The system FORM, Strong Maximum Test, copula/joint distribution, DDO/LQI,
formatting, and SORM prerequisite fixes are integrated into ``v2.0``. Their
v1.x PRs remain under separate maintainer control; integrating their commits
does not retarget or close those PRs.

The corrected reference point is the Git tag ``baseline/v2.0-before-naming``.
All 478 tests and all 13 indexed tutorials pass before and after the naming
pass. Canonicalizing the mapped identifiers gives matching Python syntax trees
across all 45 package modules, apart from documentation and version metadata.
No numerical formulas or tolerances changed in the naming commit.

A follow-up replaces wildcard imports in the distribution package to correct
the module collisions described above. Five import regression cases bring the
suite to 483 tests at that checkpoint. The calibration replacement adds direct
analytic and failure regressions while preserving all nine specialist reference
tests. Both original 10-by-10 generic tutorial grids match to floating-point
precision. See the migration plan for the latest validation record.

The SORM prerequisite fix is included. Monte Carlo also now reports an infinite
estimated reliability index when no failures are sampled, consistent with its
zero probability estimate. This is not evidence of zero true failure
probability; finite-sample uncertainty reporting remains further work.

Repository migration tools
--------------------------

``scripts/migrate_names.py`` applies the reviewed map to tracked repository
code and notebook source, with a dry run by default. It preserves notebook
outputs and ignores untracked files. It is a repository maintenance tool,
not a type-aware converter for arbitrary user scripts.

``scripts/api_inventory.py --check-names`` checks package function/method names
and the migrated class names. ``--check-migration`` verifies baseline manifest
coverage and current calibration definition/signature records.
``scripts/execute_notebooks.py --output-dir DIR``
executes the tutorials listed in the tutorial index in fresh kernels. CI runs
these checks, the test suite, formatting, documentation, and package builds.

Active learning development branch
----------------------------------

The previously unmerged ``al`` work is integrated into 2.0. Import from
``pystra.active_learning``. ``PceSurrogate``, ``learning_u`` and
``learning_eff`` follow the 2.0 naming conventions. Surrogates now consume
independent normal coordinates, not physical points.

``ActiveLearning.run()`` returns an immutable ``ActiveLearningResult`` with
``failure_probability``, ``beta``, conditional sampling diagnostics and explicit
convergence status. It replaces the development branch's getters and mutable
``Pf``/history fields. Settings are keyword-only; use lowercase ``u`` or ``eff``
for the named learning functions. The former development branch's implicit
beta-stability logic is replaced by explicit stopping policies.

The four components are now ``Surrogate``, ``LearningFunction``,
``ReliabilityEstimator`` and ``StoppingCriterion``. The import path remains
``pystra.active_learning``, now a package. Existing concise calls retain the
same default workflow and numerical settings: ``learning_threshold`` configures
the named U/EFF policy, ``n_estimation`` configures final independent MC, and
``target_cov`` configures ``LearningThreshold``. For explicit components use
``UFunction(threshold=...)``, ``MonteCarloEstimator(n_samples=...)`` and
``LearningThreshold(target_cov=...)`` instead. Combining a component with its
shortcut raises ValueError. Configure these objects directly rather than
assigning former runner attributes ``learning_threshold``, ``n_estimation``
or ``target_cov``. ``analysis.learning_function`` now holds a policy object.

``BetaBounds``, ``BetaStability`` and ``AllCriteria`` are explicit alternatives.
History entries add ``probability_band``, ``beta_band`` and
``learning_satisfied``. The result now stores a single immutable ``estimate``
record. Attribute access to ``failure_probability``, ``beta``, ``sampling_cov``,
``sampling_interval`` and ``n_estimation`` is unchanged, but these are properties;
``dataclasses.asdict(result)`` nests sampling data under ``estimate``. Use named
fields when constructing result records; their positional constructor has
changed. A custom estimator may return no sampling interval (None), and
``result.estimate`` identifies its method, dependence and confidence level.

``seed`` controls run-owned randomness. Explicit surrogate instances retain
their own optimizer/bootstrap configuration; set their seed at construction.
See :doc:`active_learning` for composition and extension contracts, and
:doc:`notebooks/ex_active_learning` for scope and limitations.

PCE now defaults to adaptive sparse fitting: ``method="lars"`` with candidate
degrees 1 through 5. An integer ``degree`` fits a single candidate. Pass
``method="ols"`` for dense fitting. ``q_norm`` accepts one or more hyperbolic
truncations and ``max_interaction`` limits interaction order. Sparse PCE's
default initial design is ``max(30, 5*n_variables)``. ``fit_result`` returns
immutable selection diagnostics. Both PCE methods use core NumPy/SciPy;
Kriging retains the optional scikit-learn dependency.

Estimator-driven subset enrichment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SubsetSimulationEstimator`` implements the new ``EnrichmentEstimator``
contract. Pass it to ``ActiveLearning(estimator=...)`` to resample conditional
populations after each fit, then run independent final sampling. Existing
``ReliabilityEstimator`` subclasses remain final-only and retain fixed MC
pools. Their implementations need not change. ``EnrichmentResult`` separates
selection points from estimator-computed probabilities; never average its
pooled conditional samples to estimate Pf.

For adaptive estimators, ``n_candidates`` caps the selectable pool, while
``SubsetSimulationEstimator(n_samples=...)`` sets the per-level sample size.
The default initial design is ``max(30, 5*n_variables)``. Explicit ``n_initial``
still takes precedence. Exploration uses a repeated separate random stream
for comparisons; final estimation is independent. Fixed MC seeds and results
retain their prior behaviour.

``ReliabilityEstimate`` adds ``converged``, ``status`` and ``diagnostics`` with
backward-compatible defaults for existing component constructors. Completion
of the sampling algorithm is distinct from adequate CoV. ``LearningStep`` adds
``estimation_converged`` and optional ``sampling_cov`` for exploration. Built-in
stopping rules reject incomplete exploratory estimates. The active result may
have status ``estimation_failed`` if final subset sampling cannot finish.
Subset diagnostics contain ``SubsetRun`` and ``SubsetLevel`` snapshots, with
no IID binomial confidence interval. The standalone ``pystra.SubsetSimulation``
API and numerical implementation are unchanged in this increment.

Additional active-learning contracts
------------------------------------

The public import path remains ``pystra.active_learning``. The new named
surrogate is ``"pc_kriging"`` and the new named learning function is ``"fbr"``.
``PcKrigingSurrogate`` and ``PcKrigingFitResult`` expose sequential PC-Kriging;
``ImportanceSamplingEstimator`` and ``ImportanceSamplingDiagnostics`` supply
explicit weighted sampling. Both use independent normal coordinates.

``EnsembleSurrogate.predict_replicates`` and
``EnsembleLearningFunction.select_replicates`` extend the existing scalar
contracts without changing U/EFF. ``PceSurrogate`` implements the ensemble
interface. ``FbrLearning`` requires it; incompatible combinations fail before
true-model evaluation. Named FBR defaults to ``BootstrapBounds``; existing
U/EFF defaults remain unchanged. Bootstrap probability stopping is restricted
to fixed IID normal enrichment and rejected for adaptive weighted/conditional
pools, including when nested inside ``AllCriteria``.

``LearningStep.bootstrap_probability_band`` is an optional immutable pair of
actual bootstrap Pf extremes, separate from ``probability_band`` (the
mean/spread sensitivity diagnostic). ``None`` means it was not calculated.
Neither field is a confidence interval for total surrogate error. See
:doc:`active_learning` for formulas, limitations and the worked tutorial.
