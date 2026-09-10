Migrating from 1.x to 2.0
=========================

This page describes the API currently implemented on the ``v2.0`` development
branch (``2.0.0.dev0``). Naming, the calibration replacement, and the first
FORM result contract are implemented. Remaining algorithm results, options,
and parameter changes are described in the
:download:`migration plan <../v2.0-migration-plan.md>`.

The outstanding feature PRs remain unmerged in 1.x and are reserved for 2.0.
Their functionality is new in this release, including the Strong Maximum
Test, copula/joint-distribution support, DDO/LQI refinements, and the SORM
prerequisite fix. Active learning is also new in 2.0. Earlier feature-branch
interfaces are development history, not released 1.x APIs to migrate from.

Naming conventions in 2.0
-------------------------

PySTRA 2.0 adopts PEP 8 throughout the public API. Three rules cover every
public name, and they are applied uniformly rather than case by case:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Kind of name
     - Convention
     - Example
   * - Functions, methods, parameters, attributes, modules
     - ``snake_case``
     - ``model.add_variable``, ``options.set_imax``
   * - Classes
     - ``CapWords``, with acronyms fully capitalised
     - ``SystemFORM``, ``DDOCriterion``, ``LQI``
   * - Constants
     - ``UPPER_SNAKE_CASE``
     - ``DEFAULT_BLOCK_SIZE``

The acronym rule follows PEP 8's *Descriptive: Naming Styles*, which states:
"When using acronyms in CapWords, capitalize all the letters of the acronym.
Thus ``HTTPServerError`` is better than ``HttpServerError``." Because PySTRA's
domain is dense with acronyms — FORM, SORM, PCE, PC-Kriging, FBR, DDO, LQI,
SWTP — applying this consistently is what makes the API predictable: if you
know the acronym, you know how it is spelled in the code.

Note that the rule governs *CapWords only*. Functions, methods and module
names keep their acronyms lowercase, so ``run_form`` and ``pc_kriging`` are
correct and are not affected.

``ScipyDist`` keeps its existing name. In 2.0, ``GEVmax`` remains available as
an alias for ``GEV``; it is the one deliberate alias in the new API.

Extreme-value distributions are named by family rather than by type number.
The literature labels the three Fisher–Tippett limits with Roman numerals —
Type I (Gumbel), Type II (Fréchet) and Type III (Weibull), all special cases of
the generalized extreme value (GEV) distribution — and PySTRA 1.x followed it
with names such as ``TypeIlargestValue``. In 2.0 the class takes the family
name. The bare name is the conventional flavour, maxima for ``Gumbel`` and
``Frechet`` and minima for ``Weibull``, and the opposite flavour takes a
suffix, as in ``GumbelMin`` and ``GEVMin``. ``Frechet`` is the ASCII spelling of Fréchet.

Implemented naming changes
--------------------------

Functions and methods now use snake_case. Inconsistent class names follow the
project's CapWords convention. Apart from ``GEVmax``, there are no aliases for the replaced names; old names
and module paths raise errors that name their replacements (see :ref:`signposts`).
Modules moved into subpackages (see :ref:`moved-modules`). Calibration is now a package
with explicit exports; its structural changes are described below.

.. _renamed-classes:

Renamed classes
~~~~~~~~~~~~~~~

The classes below were renamed in 2.0. Apart from ``GEVmax``, which remains an
alias for ``GEV``, there are no aliases; an old name raises an error that names its replacement
(see :ref:`signposts`). ``TypeIlargestValue`` and ``TypeIIIsmallestValue`` duplicated
``Gumbel`` and ``Weibull`` exactly, so those classes are removed and the
existing names used instead.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - 1.x
     - 2.0
   * - ``Form``
     - ``FORM``
   * - ``Sorm``
     - ``SORM``
   * - ``TypeIlargestValue``
     - ``Gumbel``
   * - ``TypeIsmallestValue``
     - ``GumbelMin``
   * - ``TypeIIlargestValue``
     - ``Frechet``
   * - ``TypeIIIsmallestValue``
     - ``Weibull``
   * - ``GEVmax``
     - ``GEV`` (``GEVmax`` still works as an alias)
   * - ``GEVmin``
     - ``GEVMin``
   * - ``KofNSystem``
     - ``KOfNSystem``
   * - ``GenericModel``
     - ``NormalizedReliabilityModel``
   * - ``GenericCalibration``
     - ``CodeCalibration``

Classes added during 2.0 development follow the same rule: ``FORMResult``,
``PCESurrogate``, ``PCECandidate``, ``PCEFitResult``, ``PCKrigingSurrogate``,
``PCKrigingFitResult`` and ``FBRLearning``. Earlier development snapshots
spelled them in title case (``FormResult``, ``PceSurrogate`` and so on) and
numbered the extreme-value classes (``Type1LargestValue`` and so on); those
spellings were never released.

.. _moved-modules:

Moved modules
~~~~~~~~~~~~~

Modules are grouped into subpackages in 2.0. Code that imports from the
top-level ``pystra`` namespace is unaffected; code that imports a module path
directly needs the new path.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 1.x module
     - 2.0 module
   * - ``pystra.analysis``
     - ``pystra.reliability.analysis``
   * - ``pystra.form``
     - ``pystra.reliability.form``
   * - ``pystra.sorm``
     - ``pystra.reliability.sorm``
   * - ``pystra.mc``
     - ``pystra.reliability.monte_carlo``; ``ImportanceSampling`` is in
       ``pystra.reliability.importance_sampling``
   * - ``pystra.ls``
     - ``pystra.reliability.line_sampling``
   * - ``pystra.ss``
     - ``pystra.reliability.subset_simulation``
   * - ``pystra.sensitivity``
     - ``pystra.reliability.sensitivity``
   * - ``pystra.correlation``
     - ``pystra.dependence.correlation``
   * - ``pystra.transformation``
     - ``pystra.dependence.transformation``
   * - ``pystra.loadcomb``
     - ``pystra.loads``
   * - ``pystra.integration``, ``pystra.quadrature``,
       ``pystra.cholesky_sensitivity``
     - Private numerical helpers in ``pystra._numerics``; no longer public

Modules added during 2.0 development moved with them: ``copula`` and ``joint``
to ``pystra.dependence``; ``system_form`` and ``strong_maximum`` to
``pystra.reliability``; ``system`` to ``pystra.systems``; ``ddo`` to
``pystra.decision``; and ``fbc`` into ``pystra.loads``. Distribution modules with compound names now use
snake_case, for example ``pystra.distributions.shifted_lognormal``; import
the classes themselves from ``pystra`` or ``pystra.distributions``.

.. _top-level-namespace:

Top-level namespace
~~~~~~~~~~~~~~~~~~~

``import pystra`` gives the everyday modelling classes, distributions,
dependence models, reliability methods, systems, load processes and result
types, options and the error classes: 70 names, listed in ``pystra.__all__``. Every module declares its own
``__all__``, and nothing else leaks into the namespace. Specialised workflow
tools are imported from their subpackage:

- code calibration: ``pystra.calibration`` (``CodeCalibration``,
  ``solve_designs``, ``derive_factors`` and so on);
- design decisions and target reliability: ``pystra.decision`` (``DDO``,
  ``LQI``, ``SWTP``, ``TargetReliability`` and so on);
- active-learning components: ``pystra.active_learning``. ``ActiveLearning``
  itself is also available as ``pystra.ActiveLearning``.

These tools are new in 2.0. Using one of their names at the top level raises
an error that names the subpackage to import it from.

.. _signposts:

Signposts for old names
~~~~~~~~~~~~~~~~~~~~~~~

Apart from ``GEVmax``, 2.0 keeps no aliases for replaced names, but it tells you
where each one went:

- ``pystra.Form`` and every other 1.x name raise an ``AttributeError`` that
  names the replacement, for example "PySTRA 2.0 renamed Form to FORM: use
  pystra.FORM".
- ``import pystra.form`` and ``from pystra.form import ...`` raise an
  ``ImportError`` that names the new module.
- ``from pystra import Form`` raises Python's own ``ImportError``: Python
  replaces a package's message with its generic one for this form of import.
  It adds a "Did you mean" suggestion when the new name is similar, as it is
  for ``FORM``.

The messages are generated from the migration records by
``scripts/generate_signposts.py``. They will be removed in 3.0.

.. _errors:

Errors
~~~~~~

PySTRA raises its own exception types. An invalid model, distribution or input
specification raises ``pystra.ModelError``, which is also a ``ValueError``; in 1.x
these checks raised a bare ``Exception``. An analysis that cannot produce a valid
result raises ``pystra.AnalysisError``, which is also a ``RuntimeError``. Both
derive from ``pystra.PystraError``.

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

The class names ``SystemFORM``, ``FBCProcess``,
``ScipyDist``, ``DDO``, ``DDOCriterion``, ``DDOObjective``, ``LQI``, ``SWTP``,
``SWTPRecord``, and ``SWTPIndexRecord`` retain their original spellings in the
source and feature contributions. This does not imply that every class was
available in a published 1.x release. Acronyms in
CapWords retain their capitals, as recommended by
`PEP 8 <https://peps.python.org/pep-0008/#descriptive-naming-styles>`_.
``Form`` and ``Sorm`` became ``FORM`` and ``SORM`` under the same rule, and
the extreme-value classes are now named by family (see :ref:`renamed-classes`).

The :download:`naming map <../migration/naming-map.json>` records the
reviewed spelling changes and retained class names. The :download:`definition manifest <../migration/api-migration.json>`
maps all 645 inventoried definitions, including private helpers. The
:download:`baseline inventory <../migration/api-baseline.json>` records
signatures, assigned attributes, dependency versions, and observed exports.
Inventory inclusion does not imply that an internal helper or an incidental
third-party export is a supported public API.
The baseline includes unmerged feature PRs; its ``1.6.0`` version label does
not make their interfaces part of the released 1.x API. For new joint
distributions, use ``joint.make_transformation(...)`` directly.

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
    form = ra.FORM(
        stochastic_model=model,
        limit_state=ra.LimitState(lambda R, S: R - S),
    )
    result = form.run()
    print(result.beta, result.converged)

The names ``stdv``, ``stochastic_model``, and the existing solver getters are
transitional. Scientific acronyms such as FORM and LQI retain their
conventional spelling in prose.

Options and constructors
------------------------

``AnalysisOptions`` is removed. Each analysis takes one frozen settings object
through its ``options`` keyword:

.. list-table::
   :header-rows: 1

   * - Settings
     - Used by
   * - ``FORMOptions``
     - ``FORM``, ``SystemFORM``, ``SensitivityAnalysis``, ``StrongMaximumTest``
   * - ``SORMOptions``
     - ``SORM``
   * - ``SimulationOptions``
     - ``CrudeMonteCarlo``, ``ImportanceSampling``, ``LineSampling``,
       ``SubsetSimulation``, ``DistributionAnalysis``, ``ActiveLearning``

Unknown fields are rejected, values are validated when the object is created,
and the defaults are those of 1.x. Derive modified settings with
:func:`dataclasses.replace`. A simulation method rejects a setting that it
does not use, such as ``target_cov`` for line sampling.

.. list-table:: Settings
   :header-rows: 1

   * - 1.x
     - 2.0
   * - ``set_imax(n)``
     - ``FORMOptions(max_iterations=n)``
   * - ``set_e1(e)``, ``set_e2(e)``
     - ``limit_state_tolerance``, ``gradient_tolerance``
   * - ``set_step_size(s)``
     - ``step_size``
   * - ``set_diff_mode("ddm")``
     - ``differentiation="ddm"``
   * - ``set_ffd_parameter(p)``
     - ``ffd_parameter`` (FORM and SORM)
   * - ``set_block_size(n)``
     - ``block_size`` (FORM and simulation)
   * - ``set_samples(n)``
     - ``SimulationOptions(n_samples=n)``
   * - ``target_cov``, ``stdv_sim``, ``set_bins(n)``
     - ``target_cov``, ``sampling_std``, ``bins``
   * - ``set_transform(t)``, ``set_rosenblatt_order(o)``
     - ``transform``, ``rosenblatt_order``
   * - ``set_print_output(flag)``
     - Removed: analyses do not print; use ``result.summary()``
   * - ``SORM.run(fit_type="pf")``
     - ``SORMOptions(fit="point")``; ``fit_type="cf"`` is ``fit="curve"``

``transf_type``, ``Ro_method``, ``flag_sens``, ``Recorded_u``, ``Recorded_x``,
``ffdpara_thetag``, ``sim_point``, ``multi_proc`` and ``random_generator`` had
no effect and are removed.

Constructors take the model and limit state first, then keyword-only
settings: ``FORM(model, limit_state, *, options=None)``.

.. code-block:: python

    # 1.x
    options = ra.AnalysisOptions()
    options.set_imax(50)
    form = ra.FORM(stochastic_model=model, limit_state=limit_state,
                   analysis_options=options)

    # 2.0
    form = ra.FORM(model, limit_state, options=ra.FORMOptions(max_iterations=50))

``SensitivityAnalysis(model, limit_state, *, options=None, method="numerical",
delta=0.01)`` takes the method and step that were ``run()`` arguments; use
``method="closed_form"`` for ``run(numerical=False)``. ``SystemFORM(model,
system, ...)`` takes the model first. Both reverse the 1.x argument order, and
``run_form()`` is removed. The Monte Carlo classes no longer take
``analysis_options`` first.

SORM, importance sampling and line sampling no longer run FORM when
constructed; ``run()`` does. Pass a completed analysis as ``form=`` to reuse its
design point. SORM then uses that analysis's settings and rejects a
non-default ``SORMOptions.form``. When importance or line sampling runs FORM
itself, it uses the default FORM settings with its own block size and
transformation; to use others, such as direct differentiation, run FORM first
and pass it.

``LimitState.evaluate_lsf(x, model, *, differentiation="no",
ffd_parameter=1000, block_size=1000)`` takes explicit settings instead of an
options object.

Randomness
----------

Crude Monte Carlo, importance sampling, distribution analysis, line sampling
and subset simulation drew from NumPy's global random state, so
``np.random.seed`` controlled them. They now take ``rng``, as do
``StrongMaximumTest`` and ``ActiveLearning``, where it replaces ``seed``:

.. code-block:: python

    # 1.x
    np.random.seed(2026)
    analysis = ra.CrudeMonteCarlo(...)

    # 2.0
    analysis = ra.CrudeMonteCarlo(model, limit_state, rng=2026)

``rng`` accepts a seed, a ``numpy.random.Generator`` or ``None``. An integer
seed recreates the same stream on every ``run()``; a generator advances its own
state, so successive runs differ; ``None`` draws fresh entropy. The global
random state is neither used nor changed. The streams differ from 1.x, so a
seeded 1.x estimate is not reproduced exactly, but agrees within its sampling
error. A ``StrongMaximumTest`` given an integer seed now repeats its sphere
sample on each run instead of advancing.

Result records
--------------

Every analysis's ``run()`` now returns an immutable record from
:mod:`pystra.results`: ``FORMResult``, ``SORMResult``, ``SimulationResult``,
``SystemFORMResult``, ``SensitivityResult``, ``StrongMaximumResult`` or
``DistributionAnalysisResult``. In 1.x only sensitivity analysis returned a
value. A later run cannot change an earlier record. Every record has
``method``, ``status``, ``message``, ``n_limit_state_evaluations`` and
``variable_names``, the ``converged`` property and ``summary()``; estimates
add ``failure_probability`` and ``beta``. See :doc:`guides/results`.

``FORMResult.beta`` is the normal-equivalent index, and ``design_index`` the
signed distance of the design point in ``standard_space``. These differ in
Student-t standard space. A normal-space index remains finite even when its
very small probability underflows to zero. Iteration exhaustion raises
``AnalysisError``, which carries the unconverged record, with no probability,
index or design point, in ``.result``; see :ref:`failure-handling`. The design point and direction are
read-only arrays in ``variable_names`` order. Records compare equal by value
but, holding arrays, are not hashable.

If you used the 2.0 development ``FORMResult``, its fields are renamed:

.. list-table::
   :header-rows: 1

   * - Before
     - Now
   * - ``design_point``
     - ``design_point_x``
   * - ``standard_point``
     - ``design_point_u``
   * - ``geometric_beta``
     - ``design_index``
   * - ``converged`` field
     - ``status`` field; ``converged`` is a property
   * - tuples
     - read-only NumPy arrays

``SensitivityAnalysis.run()`` returns a ``SensitivityResult`` instead of a
dictionary:

.. code-block:: python

    # 1.x
    result = analysis.run(numerical=True)
    result["R"]["mean"]
    closed = analysis.run(numerical=False)
    closed["marginal"]["R"]["mean"], closed["correlation"]

    # 2.0
    result = analysis.run(numerical=True)
    result.marginal["R"]["mean"]
    closed = analysis.run(numerical=False)
    closed.marginal["R"]["mean"], closed.correlation

``result.to_dataframe()`` gives the table that ``analysis.summary(result)``
gave.

Two outcomes are reported differently in the records. A simulation that
observed no failures reports an infinite coefficient of variation, rather
than crude Monte Carlo's placeholder of 1.0, and its status is
``"precision_not_met"``. A SORM fit whose curvatures make Breitung's formula
undefined printed a message and reported a probability and index of 0.0; its
record now has status ``"not_converged"`` and no estimate, and ``run()``
raises it in an ``AnalysisError`` unless ``on_failure="return"``. Likewise, an
undefined modified Breitung probability is ``None``.

Correlation matrices
--------------------

``CorrelationMatrix`` is checked when it is created: it must be square, finite,
symmetric, with a unit diagonal and positive definite, or ``ModelError`` is
raised. ``set_correlation`` checks an array in the same way; in 1.x any array
was accepted, and an invalid one failed later, if at all. The matrix is held as
a read-only array, so element assignment is removed: build a new matrix. The
unused attributes ``mu``, ``sigma`` and ``p1`` to ``p4`` are removed.

``cholesky()`` returns the lower Cholesky factor; ``nataf(model)`` returns the
standard-normal correlation that the Nataf transformation uses for the model's
marginals; and ``CorrelationMatrix.nearest_positive_definite(matrix)`` repairs
an estimate that is not positive definite, using Higham's (2002) method.

.. _failure-handling:

Failure handling
----------------

In 1.x, FORM warned when it exhausted its iterations and kept its last,
unconverged values. Nonconvergence now raises ``AnalysisError`` (a
``RuntimeError`` and a ``PystraError``), which carries the unconverged record
in ``.result``. ``FORM``, ``SORM``, ``SystemFORM`` and ``SensitivityAnalysis``
take ``on_failure="return"`` to return that record instead, with status
``"not_converged"`` and no probability, index or estimate; FORM also warns.
Use it for batch studies:

.. code-block:: python

    try:
        result = ra.FORM(model, limit_state).run()
    except ra.AnalysisError as error:
        print(error.result.limit_state_error)

    result = ra.FORM(model, limit_state, on_failure="return").run()
    if not result.converged:
        ...

SORM also fails when Breitung's formula is undefined for the fitted
curvatures, and system FORM when a component does not converge or its
integration fails. Code calibration returns its unconverged cases, as before.
Simulations do not fail: they report an unmet precision target as
``"precision_not_met"`` with their estimate. Importance and line sampling that
run FORM themselves still sample about its last point, with its warning.

Getters and printing
--------------------

The analysis getters and printing methods are removed; read the record that
``run()`` returns. The analyses' remaining numerical state (``beta``, ``Pf``,
``u``, ``kappa``, ``cov_q_bar`` and so on) is private, and ``limitstate`` is
renamed ``limit_state``.

.. list-table::
   :header-rows: 1

   * - 1.x
     - 2.0
   * - ``get_failure()``
     - ``result.failure_probability``
   * - ``get_beta()``
     - ``result.beta``; FORM's geometric index is ``result.design_index``
   * - ``get_equivalent_beta()``
     - ``result.beta``
   * - ``get_design_point()``, ``get_design_point(False)``
     - ``result.design_point_u``, ``result.design_point_x``
   * - ``get_alpha()``
     - ``result.alpha``, in ``result.variable_names`` order
   * - ``get_no_function_calls()``
     - ``result.n_limit_state_evaluations``
   * - ``show_results()``, ``show_detailed_output()``
     - ``print(result.summary())``, ``result.to_dataframe()``
   * - SORM ``pf2_breitung``, ``pf2_breitung_m``
     - ``result.approximations["breitung"]``, ``["modified_breitung"]``
   * - SORM ``kappa``, ``kappa_pf``, ``betaHL``
     - ``result.curvatures``, ``result.form.design_index``
   * - Monte Carlo ``cov_q_bar``, ``k``
     - ``result.coefficient_of_variation``, ``result.n_samples``, and
       ``result.diagnostics["history"]``
   * - Subset simulation ``thresholds``, ``conditional_probs``, ``n_levels``
     - ``result.diagnostics["thresholds"]`` and so on
   * - Strong Maximum Test ``status``, ``get_points()``, ``get_values()``
     - ``result.has_competing_points``, ``result.points(region, space)``,
       ``result.limit_state_values[result.regions[region]]``
   * - ``results_valid``
     - ``result.status``
   * - ``SensitivityAnalysis.summary(result)``
     - ``result.to_dataframe()``

FORM's getters returned the geometric index; in normal space it equals
``result.beta``, and in Student-t space use ``result.design_index``.
``plot_strong_maximum`` takes the Strong Maximum Test's record. The
distribution analysis's samples are ``result.samples_x`` and
``result.limit_state_values``; crude Monte Carlo no longer exposes its stored
samples. To test a system component's design point, pass the component limit
state and ``result.component_results[name].design_point_u`` to
``StrongMaximumTest`` explicitly.

Code calibration using normalized reliability
---------------------------------------------

``GenericCalibration`` is renamed ``CodeCalibration``, and ``GenericModel``
becomes ``NormalizedReliabilityModel``. Their implementation is in
``pystra.calibration.normalized``. These names describe the engineering
workflow and probability model; no compatibility aliases are retained.

``CodeCalibration`` is the primary code-calibration workflow. It evaluates
candidate factors against a probability model over a normalized G/P/Q load-ratio
grid. It does not silently choose a fitting objective or optimize the factors.

Replace model setters and the 14-position ``GenericModel.get()`` tuple with
explicit constructor fields. ``NominalValues`` holds characteristic values;
``CodeFactors`` holds a separate candidate factor set. Both validate positive,
finite values. ``NormalizedReliabilityModel`` copies distributions/constants and optionally
accepts a copula. The copula follows the random-variable order
resistance_error, resistance, load_error, dead_load, permanent_load, live_load,
omitting constants. Independence is the default.

For example::

    from dataclasses import replace
    import pystra as ra

    model = ra.calibration.NormalizedReliabilityModel(
        resistance=ra.Lognormal("R", 1, 0.08),
        dead_load=ra.Normal("G", 1, 0.08),
        permanent_load=ra.Normal("P", 1, 0.10),
        live_load=ra.Normal("Q", 1, 0.10),
        resistance_error=ra.Lognormal("w_R", 1, 0.05),
        load_error=ra.Lognormal("w_S", 1, 0.10),
        nominal_values=ra.calibration.NominalValues(1, 1, 1, 1),
    )
    factors = ra.calibration.CodeFactors(phi=0.8, gamma_g=1.2, gamma_p=1.5, gamma_q=1.6)
    study = ra.calibration.CodeCalibration(
        live_load_ratios=[0.2, 0.5, 0.8], dead_load_ratios=[0, 0.5, 1]
    )
    current = study.run(model, factors, target_beta=3.8)
    proposed = study.run(model, replace(factors, phi=0.9), target_beta=3.8)
    print(current.to_frame())
    fig, ax = ra.calibration.plot_calibration(
        {"Current": current, "Proposed": proposed}, target_beta=3.8
    )

This replaces ``add_model(...)``, ``analyse()`` and the study's mutable result
cache. Each ``run(model, factors, *, options=None, target_beta=None)`` returns a
new ``CodeCalibrationResult`` with the analyzed model snapshot, factors, designs,
and a ``FORMResult`` at every grid point. ``beta`` is a fresh array of shape
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

Partial and combination factor calibration
------------------------------------------

The old ``Calibration`` object is removed, without an alias. Its hidden sequence
of mutable DataFrame attributes is replaced by independently usable operations::

    problem = ra.calibration.FactorCalibrationProblem(
        cases, nominal_values=nominal_values, design_parameter="scale"
    )
    targets = ra.calibration.solve_designs(problem, target_beta=4.3, method="root")
    factors = ra.calibration.derive_factors(targets, method="matrix")
    selected = ra.calibration.select_factors(
        factors, resistance="minimum", loads="maximum", combinations="maximum"
    )
    designs = ra.calibration.design_with_factors(problem, selected)
    checks = ra.calibration.verify_designs(problem, max(designs.values), target_beta=4.3)

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

The outstanding Strong Maximum Test, copula/joint distribution, DDO/LQI,
formatting, and SORM prerequisite PRs remain unmerged in 1.x. Their commits
are included on ``v2.0`` and their changes are reserved for this release.
System FORM was included earlier. Contributor attribution is retained in
the integration records and source history.

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

Active learning: new in 2.0
---------------------------

Active learning is a new 2.0 capability, developed from the ``al`` contribution
and the literature-guided extensions. It remains unmerged in 1.x. Import from
``pystra.active_learning``. ``PCESurrogate``, ``learning_u`` and
``learning_eff`` follow the 2.0 naming conventions. Surrogates now consume
independent normal coordinates, not physical points.

``ActiveLearning.run()`` returns an immutable ``ActiveLearningResult`` with
``failure_probability``, ``beta``, conditional sampling diagnostics and explicit
convergence status. Settings are keyword-only; use lowercase ``u`` or ``eff``
for the named learning functions. Stopping policies are explicit components.

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
``PCKrigingSurrogate`` and ``PCKrigingFitResult`` expose sequential PC-Kriging;
``ImportanceSamplingEstimator`` and ``ImportanceSamplingDiagnostics`` supply
explicit weighted sampling. Both use independent normal coordinates.

``EnsembleSurrogate.predict_replicates`` and
``EnsembleLearningFunction.select_replicates`` extend the existing scalar
contracts without changing U/EFF. ``PCESurrogate`` implements the ensemble
interface. ``FBRLearning`` requires it; incompatible combinations fail before
true-model evaluation. Named FBR defaults to ``BootstrapBounds``; existing
U/EFF defaults remain unchanged. Bootstrap probability stopping is restricted
to fixed IID normal enrichment and rejected for adaptive weighted/conditional
pools, including when nested inside ``AllCriteria``.

``LearningStep.bootstrap_probability_band`` is an optional immutable pair of
actual bootstrap Pf extremes, separate from ``probability_band`` (the
mean/spread sensitivity diagnostic). ``None`` means it was not calculated.
Neither field is a confidence interval for total surrogate error. See
:doc:`active_learning` for formulas, limitations and the worked tutorial.
