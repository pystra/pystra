Design decisions and target reliability
=======================================

.. automodule:: pystra.ddo
    :no-members:

The object API below is the recommended entry point.

.. currentmodule:: pystra.ddo

.. autosummary::
    :toctree: ../gen
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

The low-level helpers and record types below back the object API.  They remain
importable from ``pystra.ddo`` for direct reproduction of the JCSS/LQI
equations, but the object API above is preferred.

.. autosummary::
    :toctree: ../gen
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
