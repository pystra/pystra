Migrating from 1.x to 2.0
========================

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

The :download:`complete naming map <../migration/naming-map.json>` records all
126 changed spellings. The :download:`definition manifest <../migration/api-migration.json>`
maps all 645 inventoried definitions, including private helpers. The
:download:`baseline inventory <../migration/api-baseline.json>` records
signatures, assigned attributes, dependency versions, and observed exports.
Inventory inclusion does not imply that an internal helper or an incidental
third-party export is a supported public API.

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
