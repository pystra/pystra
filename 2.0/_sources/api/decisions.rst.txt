Design decisions and target reliability
=======================================

.. automodule:: pystra.decision.ddo
    :no-members:

Import the object API below from ``pystra.decision``. Implementation
responsibilities are separated into
``swtp``, ``targets``, ``risk``, ``objectives``, ``criteria``, ``studies`` and
``plotting`` modules; ``ddo`` composes the studies, objectives and criteria.

``DesignStudy.run()`` returns a structured snapshot. Its table adapter retains
failed alternatives and diagnostics; ``DDO`` excludes those alternatives from
selection.

.. currentmodule:: pystra.decision

.. autosummary::
    :toctree: ../gen
    :template: custom-class-template.rst
    :nosignatures:

    ~swtp.SWTP
    ~risk.FatalityConsequence
    ~criteria.LQI
    ~targets.TargetReliability
    ~objectives.CostBenefitModel
    ~studies.DesignStudy
    ~studies.RiskStudy
    ~risk.ScenarioRiskModel
    ~risk.RiskResult
    ~ddo.DDO
    ~objectives.DDOObjective
    ~criteria.DDOCriterion
    ~targets.RackwitzTargetModel

The low-level helpers and record types below back the object API.  They remain
importable from ``pystra.decision.ddo`` for direct reproduction of the JCSS/LQI
equations, but the object API above is preferred.

.. currentmodule:: pystra.decision

.. autosummary::
    :toctree: ../gen
    :nosignatures:

    ~targets.lqi_k1
    ~targets.lqi_target_reliability
    ~targets.derive_lqi_target
    ~targets.rackwitz_table
    ~risk.jcss_lqi_risk_cost
    ~risk.jcss_lqi_risk_cost_from_result
    ~criteria.jcss_lqi_acceptability
    ~criteria.jcss_lqi_acceptability_margin
    ~criteria.jcss_lqi_is_acceptable
    ~objectives.jcss_systematic_reconstruction_objective
    ~objectives.present_value_factor
    ~objectives.annualized_safety_cost
    ~criteria.finite_difference_derivative
    ~targets.lognormal_ratio_failure_probability
    ~plotting.plot_summary
    ~targets.TargetReliabilityCalibration
    ~swtp.get_swtp
    ~swtp.get_swtp_record
    ~swtp.get_swtp_index_record
    ~swtp.index_swtp_record
    ~swtp.swtp_table
    ~swtp.SWTPRecord
    ~swtp.SWTPIndexRecord

**Use it:** :doc:`/guides/assessment` · :doc:`/notebooks/ex_design_decision_optimization` · :doc:`/theory/decisions`
