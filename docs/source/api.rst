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

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.model
    pystra.analysis
    pystra.system

Reliability Methods
-------------------

.. autosummary::
    :toctree: gen
    :template: custom-module-template.rst
    :recursive:

    pystra.form
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
    pystra.correlation
    pystra.integration
    pystra.quadrature

Load Combinations & Calibration
-------------------------------

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

    SWTP
    FatalityConsequence
    LQI
    TargetReliability
    CostBenefitModel
    DesignStudy
    RiskStudy
    ScenarioRiskModel
    RiskResult
    DDO
    DDOObjective
    DDOCriterion
    RackwitzTargetModel

The functions below are the low-level layer that backs the object API.  They
remain importable from ``pystra.ddo`` for direct reproduction of the JCSS/LQI
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
    SWTPRecord
    SWTPIndexRecord
