Migrating from 1.x to 2.0
=========================

This page describes the API currently implemented on the ``v2.0`` development
branch (``2.0.0.dev0``). The remaining result, options, parameter, and
calibration changes are described in the
:download:`migration plan <../v2.0-migration-plan.md>`.

Implemented naming changes
--------------------------

Functions and methods now use snake_case. Inconsistent class names follow the
project's CapWords convention. There are no aliases for the replaced names.
Imports through the existing modules continue to work using the new names.

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
   * - ``SystemFORM``, ``FBCProcess``
     - ``SystemForm``, ``FbcProcess``
   * - ``GEVmax``, ``GEVmin``, ``ScipyDist``
     - ``GevMax``, ``GevMin``, ``ScipyDistribution``
   * - ``DDO``, ``DDOCriterion``, ``LQI``, ``SWTP``
     - ``Ddo``, ``DdoCriterion``, ``Lqi``, ``Swtp``

The :download:`initial naming map <../migration/naming-map.json>` records the
first 126 changed spellings. The :download:`definition manifest <../migration/api-migration.json>`
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
    form.run()
    print(form.get_beta())

The names ``stdv``, ``stochastic_model``, and the getter methods above are
transitional. This first pass preserves call signatures, array shapes, method
defaults, and numerical calculations. Algorithm-specific options, returned
result objects, parameter/attribute cleanup, module organization, and the
replacement for ``Calibration`` are subsequent migration stages. Scientific
acronyms such as FORM and LQI retain their conventional spelling in prose.

Calibration naming cleanup
--------------------------

A follow-up replaces the old calibration/load-combination container prefixes
and compressed names with names describing their purpose. These changes also
affect keyword arguments, result attributes, and some helper methods. There
are no aliases for the replaced names.

.. list-table:: Calibration and load-combination names
   :header-rows: 1
   :widths: 45 55

   * - Previous name
     - Current development name
   * - ``loadcombobj``
     - ``load_combinations``
   * - ``dict_nom_vals`` / ``calibration.dict_nom``
     - ``nominal_values`` / ``calibration.nominal_values``
   * - ``calib_var``
     - ``design_parameter``
   * - ``calib_method`` / ``est_method``
     - ``design_method`` / ``factor_method``
   * - ``calibration.df_nom``
     - ``calibration.nominal_table``
   * - ``calibration.dfXstarcal``
     - ``calibration.calibrated_design_points``
   * - ``calibration.df_phi`` / ``df_gamma`` / ``df_psi``
     - ``resistance_factors`` / ``load_factors`` / ``combination_factors``
   * - ``LoadCombination(dict_dist_comb=...)``
     - ``LoadCombination(action_distributions=...)``
   * - ``load_combinations.dict_dist_comb``
     - ``load_combinations.case_distributions`` (also available as ``cases``)
   * - ``dict_comb_cases``
     - ``leading_actions``
   * - ``list_dist_resist`` / ``list_dist_other`` / ``list_const``
     - ``resistance`` / ``other_variables`` / ``legacy_constants``
   * - ``lcn`` / ``label_comb_cases``
     - ``case_name`` / ``case_names``
   * - ``get_dict_dist_comb()`` / ``get_num_comb()`` / ``get_label(...)``
     - ``get_case_distributions()`` / ``get_case_count()`` / ``get_group_names(...)``

The :download:`calibration naming map <../migration/calibration-naming-map.json>`
records identifier changes and qualified parameter/attribute mappings. Apply
the initial callable/class map first when migrating from 1.x. The calibration
map distinguishes the old ``dict_dist_comb`` keyword from the attribute of the
same name: they describe different data. It is not a global text-substitution
recipe for user scripts.

This cleanup preserves calculations, table shapes, and existing constructor
paths. The action-based constructor is still transitional and deprecated;
its nested ``max``/``pit`` distributions retain their existing meaning.
Explicit ``cases=`` supports reliability evaluation but still lacks the role
metadata required by the old factor-calibration workflow. The naming changes
do not correct the ordering, failure-handling, or stale-result problems in the
:download:`calibration review <../calibration-review.md>`.

The normalized ``GenericCalibration`` workflow remains the primary direction
for the replacement API. The current factor tables are transitional results,
and this cleanup does not replace the old ``Calibration`` object. Readable
local ``df`` names, statistical degrees of freedom, and conversion methods such
as ``to_dict()`` retain their meanings.

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
current suite to 483 tests.

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
and the migrated class names. ``scripts/execute_notebooks.py --output-dir DIR``
executes the tutorials listed in the tutorial index in fresh kernels. CI runs
these checks, the test suite, formatting, documentation, and package builds.
