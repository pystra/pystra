What's new in PySTRA 2.0
========================

PySTRA 2.0 is a new major version. It gives every analysis one consistent
interface, adds reliability methods and engineering workflows, and keeps its
numerics accurate far into the tails. It is not compatible with 1.x:
:doc:`migrating` shows how to update a script.

This is a pre-release (|release|). The work that remains before 2.0.0 is listed
at the end of this page.

Highlights
----------

* **One consistent interface.** Analyses take ``(model, limit_state, *,
  options=...)``, and ``run()`` returns an immutable result record with a
  common ``status``, evaluation count and ``summary()``.
* **More methods.** Active learning with Kriging, sparse PCE and PC-Kriging
  surrogates; system reliability; explicit copulas; and the Strong Maximum
  Test, alongside FORM, SORM and the simulation methods.
* **Engineering workflows.** Code calibration with normalized reliability,
  load combinations, and design-decision and target-reliability studies.
* **Accurate at high reliability.** Tail probabilities are evaluated where they
  are small, and in log space where they underflow, so transformations and
  estimators stay accurate at very large reliability indices.

A consistent interface
----------------------

* Every analysis returns a result record, such as ``FORMResult``,
  ``SORMResult`` or ``SimulationResult``. Records are immutable, share a
  ``status``, ``n_limit_state_evaluations`` and ``summary()``, and convert to
  tables with ``to_dataframe()``. FORM records give convergence diagnostics
  and the design point in physical and standard normal coordinates. See
  :doc:`guides/results`.
* Settings are frozen, validated options objects, ``FORMOptions``,
  ``SORMOptions`` and ``SimulationOptions``, which reject unknown or unused
  settings.
* Simulations take ``rng``, a seed or a NumPy ``Generator``, instead of using
  NumPy's global random state, so a seeded run repeats exactly.
* Invalid input raises ``ModelError`` and a failed analysis raises
  ``AnalysisError``. A nonconverged analysis raises unless you pass
  ``on_failure="return"``, which returns its record instead.
* Names follow PEP 8, with acronyms capitalised in class names: ``FORM``,
  ``SORM``, ``SystemFORM``. Extreme-value distributions are named by family,
  such as ``Gumbel``, ``Frechet`` and ``GEVMin``. The top-level namespace is
  curated, and old names raise errors that name their replacements.
* Distributions take native parameters as keywords, as in
  ``Gumbel("Q", loc=8.9, scale=1.56)``, and ``CorrelationMatrix`` is validated
  when it is created.

New methods
-----------

* **Active learning** (:doc:`active_learning`): Kriging with U and EFF learning,
  adaptive sparse bootstrap PCE, sequential PC-Kriging, bootstrap voting and
  active Gaussian-mixture importance sampling, with independent final sampling,
  convergence diagnostics and UQLab comparisons.
* **System reliability** (:doc:`system`): ``SystemFORM`` for series and
  parallel systems of components, composition of k-of-n, cut-set and tie-set
  systems in ``pystra.systems``, and Ditlevsen bounds.
* **Copulas and joint distributions** (:doc:`copulas`): ``JointDistribution``
  with Gaussian, Student-t, independent and Frank copulas, with Rosenblatt and
  generalized Nataf transformations, including FORM in Student-t space.
* **Strong Maximum Test** (:doc:`strong_maximum`): a post-FORM check for
  competing design points.

Engineering workflows
---------------------

* **Code calibration** (:doc:`guides/calibration`): target solving, factor
  derivation, selection, design and verification as explicit operations with
  normalized reliability, replacing the stateful ``Calibration`` class.
* **Assessment** (:doc:`guides/assessment`) and load combinations with explicit
  cases and roles.
* **Decisions**: design-decision optimisation, the life quality index,
  societal willingness to pay and target reliability, in ``pystra.decision``.
* **Plotting** (:doc:`plotting`): helpers for common reliability figures.

Numerical accuracy
------------------

* **High reliability** (:doc:`guides/high_reliability`): every distribution
  provides ``sf``, ``isf``, ``logcdf``, ``logsf`` and ``logpdf``, and the
  transformations evaluate each tail where its probability is small. SORM,
  importance sampling and line sampling keep a finite reliability index when
  the probability underflows.
* **Robust runs**: nonfinite limit-state values and failing external solvers
  raise ``AnalysisError`` instead of becoming plausible probabilities. SORM,
  importance sampling and line sampling detect a stale FORM analysis, and
  ``Maximum`` and ``MaxParent`` compute their moments deterministically.
* **Corrections**: finite differences no longer shift SORM's gradient points;
  SORM handles design directions aligned with a coordinate axis; and
  ``GEVMin`` is the correct reflection of the GEV. The :doc:`changelog` lists
  every correction whose results change.

Documentation
-------------

The documentation has a user guide organised by task, 22 executed tutorials,
including an external finite-element solver with OpenSeesPy, published
benchmark problems with their references, and theory pages for each topic.

Requirements
------------

Python 3.12 or later, NumPy, SciPy 1.11 or later, pandas and Matplotlib. The
active-learning methods need the ``al`` extra; see :doc:`install`.

Before 2.0.0
------------

The release plan still includes:

* the public contracts for batched evaluators and for extending distributions;
* restructuring of the decision and assessment modules;
* a converter for user scripts, and migration trials with representative
  scripts during a public beta;
* testing on Windows and macOS and with minimum and current dependencies,
  performance baselines and release automation;
* a release candidate with a frozen API.
