API Reference
=============

This section documents the full public API of Pystra.  Every class,
function, and attribute listed below is generated automatically from
the source-code docstrings using Sphinx ``autosummary``.

Core Framework
--------------

.. toctree::
    :maxdepth: 1

    system
    strong_maximum
    copulas

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.model
    pystra.analysis
    pystra.system
    pystra.system_form

Reliability Methods
-------------------

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.form
    pystra.results
    pystra.strong_maximum
    pystra.sorm
    pystra.mc
    pystra.ls
    pystra.ss
    pystra.sensitivity

Probability Transformation
--------------------------

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.transformation
    pystra.copula
    pystra.joint
    pystra.correlation
    pystra.integration
    pystra.quadrature

Load Combinations & Calibration
-------------------------------

Start with ``GenericModel``, ``CodeFactors`` and ``GenericCalibration.run`` for
normalized code studies; ``plot_calibration`` consumes completed results.
Specialist design-point methods are separate operations in
``pystra.calibration.factors``. See :doc:`migrating` for their sequence and
assumptions, and the calibration tutorials for worked examples.

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.fbc
    pystra.loadcomb
    pystra.calibration

Distributions
-------------

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.distributions

Design Decision Optimization
----------------------------

.. automodule:: pystra.ddo
    :no-members:

The object API below is the recommended entry point.

.. currentmodule:: pystra.ddo

.. autosummary::
    :toctree: gen
    :template: custom-class-template.rst
    :nosignatures:

    Swtp
    FatalityConsequence
    Lqi
    TargetReliability
    CostBenefitModel
    DesignStudy
    RiskStudy
    ScenarioRiskModel
    RiskResult
    Ddo
    DdoObjective
    DdoCriterion
    RackwitzTargetModel

The low-level helpers and record types below back the object API.  They remain
importable from ``pystra.ddo`` for direct reproduction of the JCSS/LQI
equations, but the object API above is preferred.

.. autosummary::
    :toctree: gen
    :nosignatures:

    lqi_k1
    lqi_target_reliability
    derive_lqi_target
    rackwitz_table
    jcss_lqi_risk_cost
    jcss_lqi_risk_cost_from_result
    jcss_lqi_acceptability
    jcss_lqi_acceptability_margin
    jcss_lqi_is_acceptable
    jcss_systematic_reconstruction_objective
    present_value_factor
    annualized_safety_cost
    finite_difference_derivative
    lognormal_ratio_failure_probability
    plot_summary
    TargetReliabilityCalibration
    get_swtp
    get_swtp_record
    get_swtp_index_record
    index_swtp_record
    swtp_table
    SwtpRecord
    SwtpIndexRecord
