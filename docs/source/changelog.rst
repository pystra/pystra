Changelog
=========

All notable changes to Pystra are documented here.

The format follows `Keep a Changelog <https://keepachangelog.com/>`_.

Unreleased
----------

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
--------------------

Added
~~~~~
- Generic calibration model.
- Documentation equation display fixes.

Changed
~~~~~~~
- Tidied tutorial notebooks.


v1.2.3 (2023-09-01)
--------------------

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
--------------------

Added
~~~~~
- User-defined correlation and analysis options in load combinations.

Fixed
~~~~~
- Minor docstring and import fixes.


v1.2.1 (2023-04-01)
--------------------

Fixed
~~~~~
- Docstring rendering for GitHub Pages build.
- Syntax highlighting in notebook examples.


v1.2.0 (2023-02-01)
--------------------

Added
~~~~~
- ``Calibration`` class for partial and combination factor calibration.
- ``LoadCombination`` class for load combination reliability analysis.
- ``ZeroInflated`` distribution.
- Load combination tutorial notebooks.
- Pandas added as a dependency.


v1.1.1 (2022-01-01)
--------------------

Fixed
~~~~~
- Minor fixes and documentation updates.


v1.1.0 (2021-09-01)
--------------------

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
--------------------

Initial release.

- FORM and SORM reliability methods.
- Crude Monte Carlo simulation.
- Nataf isoprobabilistic transformation.
- 15+ probability distributions.
- Stochastic model and limit state function framework.
