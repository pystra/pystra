Changelog
=========

All notable changes to PySTRA are documented here.

The format follows `Keep a Changelog <https://keepachangelog.com/>`_.

2.0.0 (unreleased)
------------------

The outstanding feature PRs remain unmerged in 1.x. Their functionality and
fixes are new for 2.0, alongside the breaking API migration described in
:doc:`migrating`.

Added
~~~~~
- Active-learning reliability from the ``al`` contribution: Kriging with
  U/EFF learning, adaptive sparse bootstrap Hermite PCE, independent final
  sampling, explicit convergence diagnostics and benchmark tutorials.
- Adaptive sparse PCE based on UQLab 2.2.0 hybrid-LARS and corrected-LOO
  selection across polynomial degrees and hyperbolic truncations. Includes
  fit diagnostics, executed UQLab comparisons, retained BSD notices and
  uniform bootstrap resampling.
- DDO/LQI refinements from PR #93, including decision-study workflows,
  target-reliability calculations, source attribution and worked examples.
- Sequential PC-Kriging with GLS trend uncertainty and explicit correlation
  choices; bootstrap voting/FBR with actual probability-range stopping; and
  active Gaussian-mixture importance sampling with weighted diagnostics.
  Includes UQLab numerical comparisons and an executed composition tutorial.
- Worked Rosenblatt-ordering tutorial reproducing Meinen and Steenbergen's
  (2025) system example, with transformation geometry, coordinate alignment,
  and original-event integration/simulation checks.
- OpenSeesPy tutorial: a portal frame analysed by an external finite-element
  solver, with FORM and SORM checked by direct simulation of the frame.
- Every analysis's ``run()`` returns an immutable record: ``FORMResult``,
  ``SORMResult``, ``SimulationResult``, ``SystemFORMResult``,
  ``SensitivityResult``, ``StrongMaximumResult`` or
  ``DistributionAnalysisResult``, with a common ``status``, evaluation count
  (``n_limit_state_evaluations``, shared by ``ActiveLearningResult``) and
  ``summary()``. FORM records include convergence diagnostics, the
  normal-equivalent index, the design index and the design point in physical
  and standard coordinates.
- Explicit code factors, nominal values and isolated normalized-reliability study
  results, with separate plotting and optional copula dependence.
- Executed copula/transformation and Strong Maximum Test tutorial notebooks,
  with derivations and validation examples in the theoretical background.
- ``JointDistribution`` separates continuous marginals and copula dependence:
  Gaussian, Student-t, independent and bivariate Frank copulas, with densities,
  CDFs, sampling, normal Rosenblatt and elliptical Nataf transformations.
- FORM supports spherical Student-t Nataf space with the Student-t tail;
  ``get_equivalent_beta()`` returns the normal-equivalent probability index.
- ``StrongMaximumTest`` for optional post-FORM sphere diagnostics, with
  reproducible sampling, explicit budgets and candidate point groups.
- ``SystemFORM`` for component-based series/parallel reliability, joint normal
  probabilities, component diagnostics and bounds on the linearized event.
- ``pystra.systems`` module for composing named component limit states into
  nested series, parallel, k-of-n, cut-set, and tie-set system limit states.
- ``pystra.ditlevsen_bounds`` for second-order bounds from component event
  probabilities and pairwise intersections.
- System reliability documentation covering scope, usage, and the split
  between physical-space system composition and isoprobabilistic
  transformations.
- System reliability tutorial notebook and theory content with classical
  benchmark references.
- Every distribution provides ``sf``, ``isf``, ``logcdf``, ``logsf`` and
  ``logpdf``, each accurate in its own tail. Marginal transformations evaluate
  the tail that a point lies in and use log probabilities where probabilities
  underflow, so they stay finite and accurate to :math:`|u| \approx 37.5`, and
  beyond that for the normal, lognormal and Gumbel families. The
  :doc:`guides/high_reliability` guide shows what this covers in practice.
- ``LimitState.evaluate`` evaluates a point ``(n_variables,)`` or a batch of
  rows ``(n_samples, n_variables)``, with forward-difference or analytic
  gradients. Distributions expose ``parameters`` and ``with_parameters`` to
  rebuild them, and transformations provide ``jacobian_u_wrt_x`` and
  ``jacobian_x_wrt_u``.
- ``python -m pystra.migrate`` converts 1.x scripts and notebook code
  conservatively, with dry-run diffs and reports of the changes that need
  judgment. Migration trials on the 1.6.0 examples and tutorials record their
  numerical agreement.
- User guides for the extreme-value families, with their classical Type I, II
  and III names, and for reliability sensitivity analysis.

Fixed
~~~~~
- Forward finite differences modified the evaluation points they were given.
  FORM's reported results were unaffected, but SORM computed its gradient
  transformation at points shifted by the difference step (standard deviation
  / 1000 in each variable), which biased its curvatures. For the FERUM example,
  the Breitung index moves from 3.85376 to 3.85387, in line with an independent
  u-space central-difference calculation (3.85389).
- Distributions reject a zero or negative standard deviation, as their error
  message always said; previously only an infinite value was rejected.
- Calibration now respects named case roles and design parameters, rejects
  failed target solves, and retains failed generic-study points without stale
  cached results. Factor assembly is independent of case/variable ordering.
- Monte Carlo's estimated reliability index is infinite when no failures are
  observed, consistent with the zero probability estimate. This point estimate
  does not establish zero true failure probability from a finite sample.
- Numerical sensitivity analyses preserve the selected transformation options.
- SORM rejects unrun or nonconverged FORM analyses before curve-fitting or
  point-fitting. Failed retries clear previous SORM output, and results become
  valid only after fitting completes.
- Component limit-state adapters now filter unused shared-model variables.
- FORM distinguishes convergence from iteration exhaustion and rejects invalid
  or zero gradients. Nonconverged runs warn and have ``results_valid=False``.
- Ditlevsen bounds reject missing pairs, nonfinite inputs and intersections
  inconsistent with marginal probabilities.
- Transformations of Gumbel, Gamma, Weibull, Fréchet, GEV and other
  SciPy-based distributions returned infinite physical values above
  :math:`u \approx 8.3`, where :math:`\Phi(u)` rounds to one. Line sampling
  with a lognormal resistance and a Gumbel load now agrees with direct
  integration to four decimals at reliability indices 4.2, 6.1 and 9.0.
- ``GEVMin`` (``GEVmin`` in 1.x) took its transformation from quantiles of
  :math:`-X`, returning mirrored points of the wrong sign, and converted
  between moments and location/scale with the relations for maxima:
  ``GEVMin("X", 10, 3, shape=s)`` had an actual mean of about 7.3. Results for
  models that use it change.
- ``MaxParent`` inverted its CDF by bracketing, which failed for a lognormal
  maximum with large ``N``; it now inverts through the maximum's log-CDF.
- ``Lognormal.cdf`` accepts arrays, and Student-t copula transformations use
  the upper-tail functions of composite marginals.
- Nonfinite limit-state values and failing external evaluators could become
  plausible failure probabilities, and line sampling treated failed scans as
  safe. They now raise ``AnalysisError`` with the original exception as its
  cause. A limit state whose argument names do not match the model raises
  ``ModelError``.
- Adding a random variable after ``set_correlation`` silently reset the
  correlation to identity; it now raises ``ModelError``, and correlation
  dimensions are checked.
- SORM, importance sampling and line sampling could reuse a stale FORM
  analysis after the model or limit state changed. Internally generated FORM
  is rerun on each run, a supplied FORM is checked against the problem, and
  assigning ``form`` after construction takes effect.
- SORM raised "Singular rotation matrix" for design directions aligned with a
  coordinate axis.
- ``Maximum`` and ``MaxParent`` estimated their moments from 100 global random
  draws; they now use deterministic quadrature.
- A fixed-budget simulation (``target_cov=0``) reports ``"completed"`` rather
  than a missed precision target.
- SORM, importance sampling and line sampling keep finite reliability indices
  and relative uncertainties when the failure probability underflows (tested
  to a reliability index of 40). SORM's Mills ratio no longer overflows above
  37, and importance-sampling weights no longer overflow for wide proposals in
  many dimensions.

Changed
~~~~~~~
- Renamed ``GenericCalibration`` to ``CodeCalibration`` and ``GenericModel``
  to ``NormalizedReliabilityModel``; factor derivation and verification are
  presented as operations within the same code-calibration workflow.
- Grouped tutorials and API references by purpose, separated usage guides,
  and split theory into topic pages. Updated all maintained tutorials and
  added an independent probability reference for the parabolic benchmark.
- Added ``pystra.plotting`` helpers for common reliability figures, with
  explicit coordinates, existing-axis support and diagnostic interpretation.
- Documentation builds execute changed notebooks, invalidating cached results
  when PySTRA code, notebook helpers or the Python environment changes.
- Class names follow PEP 8's acronym rule: ``FORM``, ``SORM``,
  ``FORMResult``, ``SystemFORM``, ``FBCProcess``, ``DDO``, ``LQI``, ``SWTP``,
  ``PCESurrogate``, ``PCKrigingSurrogate`` and ``FBRLearning``. Extreme-value
  distributions are named by family — ``Gumbel``, ``GumbelMin``, ``Frechet``,
  ``Weibull``, ``GEV`` and ``GEVMin`` — replacing the type-numbered classes;
  ``TypeIlargestValue`` and ``TypeIIIsmallestValue`` duplicated ``Gumbel`` and
  ``Weibull`` and are removed. ``GEVmax`` remains an alias for ``GEV``, and
  ``ScipyDist`` keeps its name. Function and method names use snake_case.
- Modules are grouped into subpackages: ``pystra.reliability``,
  ``pystra.dependence`` and ``pystra.decision``, with ``system`` renamed
  ``systems`` and low-level numerical helpers made private. Load processes and
  load combinations share ``pystra.loads``, and ``ImportanceSampling`` has its
  own module. See the migration guide for moved module paths.
- Invalid model and distribution input raises ``ModelError`` (a ``ValueError``)
  instead of a bare ``Exception``; failed analyses raise ``AnalysisError`` (a
  ``RuntimeError``). Both derive from ``PystraError``.
- Limit-state evaluation no longer stores the model, options and points on the
  ``LimitState`` or modifies the caller's points, so analyses sharing a limit
  state or model cannot interfere. Each analysis counts its own evaluations:
  ``FORM.get_no_function_calls()`` reports the current run rather than the
  model's running total.
- ``SensitivityAnalysis.run()`` returns a ``SensitivityResult`` rather than a
  dictionary; the derivatives are in ``marginal`` and ``correlation``.
- ``AnalysisOptions`` is replaced by frozen ``FORMOptions``, ``SORMOptions`` and
  ``SimulationOptions``, which validate their values and reject unknown or
  unused settings. Its nine settings that had no effect are removed, and
  analyses no longer print.
- Analysis constructors take ``(model, limit_state, *, options=None, ...)``.
  ``SensitivityAnalysis`` takes its method and step, and ``SystemFORM`` takes
  the model first. SORM, importance sampling and line sampling run FORM in
  ``run()``, not when constructed, and SORM's fit is an option.
- The simulation methods take ``rng`` (a seed, a ``Generator`` or ``None``)
  instead of using NumPy's global random state, and ``StrongMaximumTest`` and
  ``ActiveLearning`` rename ``seed`` to ``rng``. An integer seed repeats its
  stream on every run. Seeded simulation results differ from 1.x within
  sampling error.
- The analysis getters (``get_beta()``, ``get_failure()`` and so on) and printing
  methods are removed in favour of the result records, whose ``summary()`` and
  ``to_dataframe()`` replace the printed reports; the analyses' numerical state is
  private and ``limitstate`` is ``limit_state``. Crude Monte Carlo reports its
  convergence history in ``diagnostics["history"]``.
- Distributions take ``std`` and ``start_point`` (for ``stdv`` and
  ``startpoint``) and expose ``mean``, ``std`` and ``start_point`` as read-only
  properties; ``Constant`` takes ``value``; ``StochasticModel.variable(name)``
  and the read-only ``constants`` mapping replace ``get_variable()`` and
  ``get_constants()``.
- Distributions take native parameters as keyword arguments of the same
  constructor in place of ``input_type``, for example
  ``Gumbel("Q", loc=8.9, scale=1.56)`` or ``Weibull("W", scale=10, shape=2.5)``.
  The bounds of ``Weibull`` and ``Beta`` are the keywords ``lower`` and
  ``upper``, and ``start_point`` is keyword-only.
- ``CorrelationMatrix`` is validated when created (square, finite, symmetric,
  unit diagonal, positive definite) and held read-only; ``set_correlation``
  validates arrays the same way. It gains ``cholesky()``, ``nataf(model)`` and
  ``nearest_positive_definite()``.
- Nonconvergence raises ``AnalysisError`` carrying the unconverged record;
  ``FORM``, ``SORM``, ``SystemFORM`` and ``SensitivityAnalysis`` take
  ``on_failure="return"`` to return it instead, as code calibration does. In
  1.x FORM warned and kept its last values.
- In the result records, a SORM fit whose curvatures leave Breitung's formula
  undefined has no estimate and status ``not_converged``, where 1.x reported
  0.0; a simulation without failures has an infinite coefficient of variation.
- The top-level namespace is curated (70 names in ``pystra.__all__``) and every
  module declares ``__all__``. Code-calibration and decision tools, new in 2.0,
  are imported from ``pystra.calibration`` and ``pystra.decision``;
  ``ActiveLearning`` is exported at the top level. Old names and module paths
  raise errors that name their replacements.
- Removed the stateful ``Calibration`` class in favor of explicit target solving,
  factor derivation, selection, design and verification operations.
- ``LoadCombination`` stores explicit cases and roles; separate evaluation and
  maximum/companion factories replace its legacy dictionary constructor.
- The ``v2.0`` development branch uses snake_case function/method names and
  consistent class names, without legacy aliases. See :doc:`migrating` for the
  implemented naming map and remaining migration stages.
- SciPy minimum version is 1.11 for multivariate Student-t CDF integration.
- Metadata and documentation consistently state GPL-3.0-or-later, with
  retained third-party notices for adapted methods.
- The column-oriented ``LimitState.evaluate_lsf`` is private; the
  transformations' ambiguous ``jacobian`` gives way to the directed methods; and
  ``parameters`` and ``with_parameters`` replace the private ``_ctor_kwargs``
  and ``_make_copy`` reconstruction hooks.

v1.6.0 (2026-03-16)
-------------------

Added
~~~~~
- **Closed-form sensitivity analysis** (Bourinet 2017): computes
  :math:`\partial\beta/\partial\theta` from a single FORM run via
  analytical differentiation of the Nataf transformation chain.
  Selected with ``SensitivityAnalysis.run(numerical=False)``.
- **Correlation sensitivities** :math:`\partial\beta/\partial\rho_{ij}`
  available from the closed-form method.
- **Generalised sensitivity parameters**: distributions can declare
  arbitrary parameters beyond mean and standard deviation via the
  ``sensitivity_params`` property (e.g. GEV shape parameter).
- ``SensitivityAnalysis.summary()`` method for DataFrame output.
- ``Distribution._make_copy()``, ``_ctor_kwargs``, and
  ``_dmoments_dtheta()`` extension points for distribution
  reconstruction and parameter differentiation.
- Cholesky differentiation module (``cholesky_sensitivity``).
- Numerical integration helpers for Nataf correlation derivatives
  (``integration`` module: ``drho0_dtheta``, ``drho_drho0``,
  ``zi_and_xi``).
- GEV (max and min) shape parameter sensitivity support.
- Sensitivity analysis notebook with worked Bourinet (2017) examples.
- Developer guide: new section on adding distributions with
  sensitivity support.
- Theory docs: generalised parameter support section.
- Theory docs: expanded SORM section with quadratic approximation
  derivation, curvature computation, Hohenbichler–Rackwitz motivation,
  and curve-fitting vs point-fitting comparison.

Fixed
~~~~~
- GEVmin ``_make_copy`` roundtrip: moments are now correctly
  preserved across reconstruction.
- ``dF_dtheta`` handles array inputs from quadrature grids.
- Sphinx documentation build: all warnings resolved.
- Console output separator width corrected (``n_hyphen`` 54 → 58).
- **Gumbel** ``input_type`` parameter: native Gumbel parameters
  (μ, β) now correctly interpreted as location and scale (issue #67).
- **Monte Carlo** ``cov_of_q_bar`` typo: fixed ``AttributeError``
  when the coefficient of variation is exactly zero (issue #64).

Changed
~~~~~~~
- Sensitivity result dicts now keyed by each distribution's declared
  ``sensitivity_params`` (backward-compatible for standard
  distributions).
- Test suite expanded to 302 tests.


v1.4.0 (2026-03-13)
-------------------

Added
~~~~~
- **SORM point-fitting** (``fit_type='pf'``): alternative to
  curve-fitting that locates fitting points on the failure surface via
  Newton iteration, yielding asymmetric curvatures.  Based on
  Henry Nguyen's contribution (PR #65).
- ``@property`` accessors on ``StochasticModel`` for ``constants``,
  ``names``, ``n_marg``, ``marginal_distributions``, ``correlation``,
  ``modified_correlation``, and ``call_function`` (backward-compatible
  with existing getter methods).
- SVD-based isoprobabilistic transformation as an alternative to
  Cholesky (``Transformation(transform_type="svd")``).
- Comprehensive test suite: 252 tests covering distributions,
  model, transformation, sensitivity, and numerics (up from 19).
- CHANGELOG, improved docstrings across all modules.

Fixed
~~~~~
- **Type II Largest Value distribution**: corrected ``invweibull``
  parametrisation (``c=k`` instead of ``c=-k-2``), which previously
  produced ``NaN`` moments.
- **ZeroInflated**: ``set_zero_probability()`` now correctly updates
  the complement probability ``q = 1 - p``.
- **Transformation**: error handling re-raises ``LinAlgError`` instead
  of printing and silently continuing.
- **NumPy 2.0+ compatibility**: fixed read-only arrays from
  ``np.eye()``, scalar/array coercion deprecation in the Jacobian,
  and column-vector inputs in ``x_to_u`` / ``u_to_x``.
- **Pandas compatibility**: fixed read-only ``DataFrame.values`` in
  calibration module.

Changed
~~~~~~~
- Build dependency changed from ``oldest-supported-numpy`` to
  ``numpy``.
- CI matrix updated: Python 3.9 -- 3.13 (dropped 3.8).


v1.3.0 (2024-04-01)
-------------------

Added
~~~~~
- Generic calibration model.
- Documentation equation display fixes.

Changed
~~~~~~~
- Tidied tutorial notebooks.


v1.2.3 (2023-09-01)
-------------------

Added
~~~~~
- Correlation support in ``LoadCombination``.
- Print correlation DataFrame.

Fixed
~~~~~
- ``Calibration`` psi factor estimation via matrix method.

Changed
~~~~~~~
- Design now uses min phi and max psi by default.
- Updated build system to ``pyproject.toml``.


v1.2.2 (2023-06-01)
-------------------

Added
~~~~~
- User-defined correlation and analysis options in load combinations.

Fixed
~~~~~
- Minor docstring and import fixes.


v1.2.1 (2023-04-01)
-------------------

Fixed
~~~~~
- Docstring rendering for GitHub Pages build.
- Syntax highlighting in notebook examples.


v1.2.0 (2023-02-01)
-------------------

Added
~~~~~
- ``Calibration`` class for partial and combination factor calibration.
- ``LoadCombination`` class for load combination reliability analysis.
- ``ZeroInflated`` distribution.
- Load combination tutorial notebooks.
- Pandas added as a dependency.


v1.1.1 (2022-01-01)
-------------------

Fixed
~~~~~
- Minor fixes and documentation updates.


v1.1.0 (2021-09-01)
-------------------

Added
~~~~~
- ``SensitivityAnalysis`` class.
- ``DistributionAnalysis`` class.
- ``ImportanceSampling`` Monte Carlo method.
- Sensitivity analysis tutorial notebook.

Changed
~~~~~~~
- Improved distribution interface.


v1.0.0 (2021-06-01)
-------------------

Initial release.

- FORM and SORM reliability methods.
- Crude Monte Carlo simulation.
- Nataf isoprobabilistic transformation.
- 15+ probability distributions.
- Stochastic model and limit state function framework.
