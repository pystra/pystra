import numpy as np
import pandas as pd
import pytest

import pystra as ra


def test_ddo_namespace_is_explicit():
    # Canonical object surface is promoted to the package top level.
    assert hasattr(ra, "ddo")
    assert hasattr(ra, "Ddo")
    assert hasattr(ra, "DdoCriterion")
    assert hasattr(ra, "DdoObjective")
    assert hasattr(ra, "Lqi")
    assert hasattr(ra, "Swtp")
    assert hasattr(ra, "TargetReliability")
    assert hasattr(ra, "RackwitzTargetModel")
    assert hasattr(ra, "RiskStudy")
    assert hasattr(ra, "ScenarioRiskModel")

    # Low-level helpers stay importable from the module but off the top level.
    assert not hasattr(ra, "plot_summary")
    assert not hasattr(ra, "TargetReliabilityCalibration")
    assert hasattr(ra.ddo, "plot_summary")
    assert hasattr(ra.ddo, "TargetReliabilityCalibration")
    assert hasattr(ra.ddo, "lqi_k1")
    assert hasattr(ra.ddo, "jcss_lqi_risk_cost")


def test_swtp_from_lqi_and_country_lookup():
    swtp = ra.ddo.Swtp.from_lqi(
        gross_domestic_product_per_capita=25010,
        work_leisure_parameter=0.16,
        demographic_constant=16,
        currency="PPPUSD",
        price_year=1999,
    )

    assert swtp.value_per_life == pytest.approx(2_501_000)
    assert ra.ddo.get_swtp("DE", indexed=False) == pytest.approx(1_900_000)
    assert ra.ddo.get_swtp("UK", indexed=False) == pytest.approx(1_700_000)
    assert ra.ddo.get_swtp("US", indexed=True) == pytest.approx(5_220_864, rel=1e-4)
    assert ra.ddo.get_swtp_record("New Zealand", indexed=False).code == "NZ"
    # The anchor is never returned silently: indexed must be explicit.
    with pytest.raises(ValueError, match="indexed=True or indexed=False"):
        ra.ddo.get_swtp("DE")
    with pytest.raises(ValueError, match="indexed=True or indexed=False"):
        ra.ddo.get_swtp_record("DE")
    assert ra.ddo.swtp_table(indexed=False).loc[
        "NL", "value_per_life"
    ] == pytest.approx(2_800_000)
    indexed = ra.ddo.swtp_table(indexed=True)
    assert indexed.loc["CH", "price_year"] == 2024
    assert indexed.loc["CH", "index_factor"] == pytest.approx(2.7775, rel=1e-4)
    # The anchor table is never returned silently: indexed must be explicit.
    with pytest.raises(ValueError, match="indexed=True or indexed=False"):
        ra.ddo.swtp_table()


def test_lqi_target_reliability_table_and_ratio():
    k1 = ra.ddo.lqi_k1(
        safety_cost_rate=1_000,
        swtp=ra.ddo.Swtp(5_000_000),
        expected_fatalities_given_failure=2,
    )

    target = ra.ddo.lqi_target_reliability(k1)
    high_variability = ra.ddo.lqi_target_reliability(k1, variability="high")

    assert k1 == pytest.approx(1e-4)
    assert target.cost_class == "medium"
    assert target.pf == pytest.approx(1e-4)
    assert target.beta == pytest.approx(3.7)
    assert high_variability.pf == pytest.approx(5e-4)
    assert high_variability.beta < target.beta
    assert ra.Lqi.lookup_target(k1).pf == pytest.approx(target.pf)
    # Looked-up targets carry their source; calculated targets do not.
    assert "source" in target.to_dict()
    assert "source" not in ra.Lqi.derive_target(k1).to_dict()

    table = {
        "large": (1e-3, 1e-3, 3.1),
        "medium": (1e-4, 1e-4, 3.7),
        "small": (1e-5, 1e-5, 4.2),
    }
    for cost_class, (representative_k1, pf, beta) in table.items():
        target = ra.ddo.lqi_target_reliability(representative_k1)
        assert target.cost_class == cost_class
        assert target.pf == pytest.approx(pf)
        assert target.beta == pytest.approx(beta)


def test_lognormal_ratio_failure_probability():
    assert ra.ddo.lognormal_ratio_failure_probability(1.0, 0.3, 0.3) == pytest.approx(
        0.5
    )
    assert ra.ddo.lognormal_ratio_failure_probability(2.0, 0.3, 0.3) < 0.5

    with pytest.raises(ValueError, match="at least one"):
        ra.ddo.lognormal_ratio_failure_probability(1.0, 0.0, 0.0)


def test_lqi_target_can_be_derived_from_marginal_model():
    target = ra.Lqi.derive_target(1e-4, resistance_cov=0.4, load_cov=0.4)

    assert target.converged is True
    assert target.metadata["method"] == "LQI marginal"
    assert target.design == pytest.approx(7.543754537126031)
    assert target.pf == pytest.approx(1.0408151381574544e-4)
    assert target.beta == pytest.approx(3.7088982796733925)


def test_rackwitz_target_model_calibrates_reliability():
    model = ra.RackwitzTargetModel(
        safety_cost_ratio=0.03,
        failure_cost_ratio=2.5,
    )

    result = model.calibrate()

    assert isinstance(result, ra.TargetReliability)
    assert result.method == "Rackwitz/Steenbergen"
    assert result.converged is True
    assert result.metadata["method"] == "Rackwitz/Steenbergen"
    assert result.design == pytest.approx(4.837456940394679)
    assert result.pf == pytest.approx(7.320203882099348e-05)
    assert result.beta == pytest.approx(3.797090962287968)
    # Full model provenance is carried, not just the cost ratios.
    assert result.metadata["interest_rate"] == pytest.approx(0.035)
    assert {
        "base_cost",
        "obsolescence_rate",
        "load_occurrence_rate",
        "serviceability_cost_ratio",
        "demolition_cost_ratio",
        "serviceability_resistance_ratio",
        "benefit_rate",
    } <= set(result.metadata)


def test_rackwitz_objective_scales_with_base_cost():
    kwargs = dict(safety_cost_ratio=0.03, failure_cost_ratio=2.5, benefit_rate=0.1)
    unit = ra.RackwitzTargetModel(base_cost=1.0, **kwargs)
    scaled = ra.RackwitzTargetModel(base_cost=10.0, **kwargs)

    # With the annual benefit normalized to C0, every objective term scales
    # linearly with base_cost (including the benefit, which the old code did
    # not scale), and the optimizing design is unchanged.
    assert scaled.objective(5.0) == pytest.approx(10.0 * unit.objective(5.0))
    assert scaled.calibrate().design == pytest.approx(unit.calibrate().design)


def test_rackwitz_target_table_calculates_class_grid():
    table = ra.RackwitzTargetModel.table()

    # Rates and ratios that users can vary via **kwargs are reported too.
    assert {"interest_rate", "obsolescence_rate", "benefit_rate", "base_cost"} <= set(
        table.columns
    )

    assert table.shape[0] == 9
    assert list(table.columns[:7]) == [
        "relative_safety_cost",
        "failure_consequence",
        "safety_cost_ratio",
        "failure_cost_ratio",
        "design",
        "pf",
        "beta",
    ]

    moderate = table[table["failure_consequence"] == "moderate"]
    assert moderate["beta"].is_monotonic_increasing

    normal = table[table["relative_safety_cost"] == "normal"]
    assert normal["beta"].is_monotonic_increasing


def test_lqi_builds_target_from_country():
    criterion = ra.Lqi.from_country(
        "CH",
        indexed=True,
        expected_fatalities_given_failure=12,
        marginal_safety_cost=5_000,
    )

    assert criterion.swtp.price_year == 2024
    assert criterion.expected_fatalities_given_failure == pytest.approx(12)
    assert criterion.k1 == pytest.approx(5_000 / (criterion.swtp.value_per_life * 12))
    assert criterion.target.cost_class == "small"
    assert criterion.target.pf == pytest.approx(1e-5)
    assert criterion.target.beta == pytest.approx(4.2)
    assert criterion.risk_cost(1000, 1e-4) == pytest.approx(
        1000 + criterion.swtp.for_lives(12) * 1e-4
    )


def test_lqi_country_requires_explicit_index_choice():
    with pytest.raises(ValueError, match="indexed=True or indexed=False"):
        ra.Lqi.from_country(
            "CH",
            expected_fatalities_given_failure=12,
            marginal_safety_cost=5_000,
        )


def test_lqi_can_use_explicit_consequence():
    criterion = ra.Lqi.from_swtp(
        ra.Swtp(5_000_000),
        consequence=ra.FatalityConsequence(
            people_exposed=20,
            probability_death_given_failure=0.5,
        ),
        marginal_safety_cost=5_000,
    )

    assert criterion.expected_fatalities_given_failure == pytest.approx(10)
    assert criterion.k1 == pytest.approx(1e-4)


def test_swtp_country_requires_explicit_index_choice():
    with pytest.raises(ValueError, match="indexed=True or indexed=False"):
        ra.Swtp.from_country("CH")

    assert ra.Swtp.from_country("CH", indexed=False).price_year == 1999


def test_cost_benefit_model_matches_jcss_notebook_values():
    area = 85.1
    failure_probability = 2.5708544377963726e-05
    model = ra.ddo.CostBenefitModel(
        benefit_rate=1.2e4,
        interest_rate=0.02,
        service_life=100,
        construction_cost=lambda As: 5000 * As,
        failure_cost=lambda As: 5000 * As + 12 * 1.8e6 + 3e4,
    )

    assert model.present_value_factor() == pytest.approx(43.52791273148604)
    assert model.objective(area, failure_probability) == pytest.approx(
        72153.98202298091
    )
    assert model.annualized_safety_cost(area, failure_probability) == pytest.approx(
        4254.889519855429
    )


def test_canonical_jcss_lqi_acceptability_and_risk_cost():
    def failure_probability(p):
        vr = 0.2
        vs = 0.3
        numerator = np.log(p * np.sqrt((1 + vs**2) / (1 + vr**2)))
        denominator = np.sqrt(np.log((1 + vr**2) * (1 + vs**2)))
        return ra.Normal("u", 0, 1).cdf(-numerator / denominator)

    def safety_cost(p):
        return 1e6 + 1e4 * p**1.25

    margin_low = ra.ddo.jcss_lqi_acceptability(
        safety_cost=safety_cost,
        failure_rate=failure_probability,
        design=3.0,
        swtp=5e6,
        expected_fatalities_given_failure=10,
    )
    margin_high = ra.ddo.jcss_lqi_acceptability(
        safety_cost=safety_cost,
        failure_rate=failure_probability,
        design=4.4,
        swtp=5e6,
        expected_fatalities_given_failure=10,
    )

    assert margin_low < 0
    assert margin_high > 0
    assert ra.ddo.jcss_lqi_risk_cost(1000, 1e-4, ra.ddo.Swtp(5e6), 10) == pytest.approx(
        6000
    )
    assert ra.ddo.jcss_systematic_reconstruction_objective(
        benefit_rate=20_000,
        safety_cost=1_000_000,
        failure_rate=1e-4,
        failure_consequence=10_000_000,
        discount_rate=0.02,
    ) == pytest.approx(-55_000)


def test_lqi_acceptability_boundary_finds_acceptance_design():
    lqi = ra.Lqi.from_lqi(
        gross_domestic_product_per_capita=35931.0,
        work_leisure_parameter=0.175,
        demographic_constant=18.9,
        expected_fatalities_given_failure=12,
        marginal_safety_cost=5000,
    )

    def safety_cost(p):
        return 1e6 + 1e4 * p**1.25

    def failure_probability(p):
        return ra.ddo.lognormal_ratio_failure_probability(p, 0.2, 0.3)

    boundary = lqi.acceptability_boundary(
        safety_cost, failure_probability, bounds=(2.0, 6.0)
    )

    # Margin is zero at the boundary (negligible against the O(1e4)
    # derivative scale) and changes sign across it.
    assert lqi.acceptability_margin_at(
        safety_cost, failure_probability, boundary
    ) == pytest.approx(0.0, abs=1e-3)
    assert (
        lqi.acceptability_margin_at(safety_cost, failure_probability, boundary - 0.5)
        < 0
    )
    assert (
        lqi.acceptability_margin_at(safety_cost, failure_probability, boundary + 0.5)
        > 0
    )

    with pytest.raises(ValueError):
        lqi.acceptability_boundary(safety_cost, failure_probability, bounds=(2.0, 2.5))


def test_target_reliability_for_period():
    target = ra.Lqi.lookup_target(1e-4)  # annual pf 1e-4
    assert target.pf == pytest.approx(1e-4)

    independent = target.for_period(50)
    partial = target.for_period(50, dependence_interval=10)
    dependent = target.for_period(50, dependence_interval=50)

    # The conversion compounds the annual pf, not the rounded beta.
    assert independent.pf == pytest.approx(1.0 - (1.0 - 1e-4) ** 50)
    assert partial.pf == pytest.approx(1.0 - (1.0 - 1e-4) ** 5)
    # Fully dependent over the period is a single renewal: pf unchanged.
    assert dependent.pf == pytest.approx(target.pf)

    # More independent renewals -> higher period pf -> lower index.
    assert independent.beta < partial.beta < dependent.beta
    assert independent.metadata["reference_period_years"] == 50
    assert independent.method == target.method

    with pytest.raises(ValueError):
        target.for_period(0)
    with pytest.raises(ValueError):
        target.for_period(50, dependence_interval=60)


def test_target_reliability_validates_pf():
    # Boundary probabilities are allowed.
    ra.TargetReliability(pf=0.0, beta=float("inf"), method="x")
    ra.TargetReliability(pf=1.0, beta=float("-inf"), method="x")
    with pytest.raises(ValueError, match="pf must be in"):
        ra.TargetReliability(pf=1.5, beta=0.0, method="x")
    with pytest.raises(ValueError, match="pf must be in"):
        ra.TargetReliability(pf=-0.1, beta=0.0, method="x")


def test_ddo_construction_is_keyword_only():
    study = ra.ddo.DesignStudy(
        variable="As",
        values=[85.1],
        analysis=lambda area: {"pf": 2.5708544377963726e-5},
    )
    criterion = ra.Lqi.from_swtp(
        5_000_000,
        expected_fatalities_given_failure=12,
        marginal_safety_cost=5_000,
    )

    # Positional construction is rejected so study/objective/criterion cannot be
    # silently transposed.
    with pytest.raises(TypeError):
        ra.Ddo(study, criterion)


def test_ddo_summary_reports_decision_points():
    probabilities = {70.0: 7.9e-4, 85.1: 2.5708544377963726e-5, 93.0: 1.0e-6}
    study = ra.ddo.DesignStudy(
        variable="As",
        values=[70.0, 85.1, 93.0],
        analysis=lambda area: {"pf": probabilities[float(area)]},
    )
    model = ra.ddo.CostBenefitModel(
        benefit_rate=1.2e4,
        interest_rate=0.02,
        service_life=100,
        construction_cost=lambda As: 5000 * As,
        failure_cost=lambda As: 5000 * As + 12 * 1.8e6 + 3e4,
    )
    criterion = ra.Lqi.from_country(
        "CH",
        indexed=True,
        expected_fatalities_given_failure=12,
        marginal_safety_cost=5_000,
    )

    summary = ra.Ddo(study=study, objective=model, criterion=criterion).summary()

    assert list(summary["point"]) == ["economic optimum", "best feasible"]
    assert list(summary.columns) == [
        "point",
        "As",
        "pf",
        "beta",
        "objective",
        "lqi_acceptable",
    ]
    assert summary.loc[0, "As"] == pytest.approx(85.1)
    assert summary.loc[1, "As"] == pytest.approx(93.0)
    assert not summary.loc[0, "lqi_acceptable"]  # economic optimum is infeasible
    assert summary.loc[1, "lqi_acceptable"]


def test_scenario_risk_result_supports_correlated_scenarios():
    scenarios = pd.DataFrame(
        {
            "state": ["A", "B", "A+B"],
            "probability": [2.0e-3, 1.5e-3, 2.0e-4],
            "fatalities": [0.2, 0.3, 2.0],
            "economic_loss": [1.0e6, 1.5e6, 8.0e6],
            "failure": [True, True, True],
        }
    )

    risk = ra.ddo.RiskResult.from_scenarios(scenarios, failure_col="failure")
    swtp = ra.ddo.Swtp(5.0e6)

    assert risk.annual_failure_rate == pytest.approx(3.7e-3)
    assert risk.expected_fatalities == pytest.approx(0.00125)
    assert risk.expected_economic_loss == pytest.approx(5850.0)
    assert risk.life_safety_cost(swtp) == pytest.approx(6250.0)
    assert risk.total_risk_cost(swtp) == pytest.approx(12100.0)
    assert ra.ddo.jcss_lqi_risk_cost_from_result(
        safety_cost=100_000,
        risk=risk,
        swtp=swtp,
        include_economic_loss=True,
    ) == pytest.approx(112100.0)
    assert risk.scenarios.loc[
        2, "expected_economic_loss_contribution"
    ] == pytest.approx(1600.0)


def test_risk_study_evaluates_design_dependent_scenarios():
    def scenarios_for_strengthening(strengthening):
        scale = 1.0 - strengthening
        return pd.DataFrame(
            {
                "probability": np.array([2.0e-3, 1.5e-3, 2.0e-4]) * scale,
                "fatalities": [0.2, 0.3, 2.0],
                "economic_loss": [1.0e6, 1.5e6, 8.0e6],
            }
        )

    model = ra.ddo.ScenarioRiskModel(scenarios_for_strengthening)
    study = ra.ddo.RiskStudy(variable="strengthening", values=[0.0, 0.25], model=model)

    results = study.evaluate(swtp=ra.ddo.Swtp(5.0e6))

    assert list(results.columns) == [
        "strengthening",
        "annual_failure_rate",
        "pf",
        "beta",
        "expected_fatalities",
        "expected_economic_loss",
        "life_safety_cost",
        "total_risk_cost",
        "design",
    ]
    # The pf alias lets a RiskStudy feed the same DDO objects as a DesignStudy.
    assert (results["pf"] == results["annual_failure_rate"]).all()
    assert results.loc[0, "total_risk_cost"] == pytest.approx(12100.0)
    assert results.loc[1, "total_risk_cost"] == pytest.approx(9075.0)
    assert results.loc[1, "annual_failure_rate"] < results.loc[0, "annual_failure_rate"]


def test_ddo_runs_lqi_algorithm_and_selects_objective():
    probabilities = {70.0: 7.9e-4, 85.1: 2.5708544377963726e-5, 93.0: 1.0e-6}

    def analysis(area):
        pf = probabilities[float(area)]
        return {"pf": pf}

    study = ra.ddo.DesignStudy(
        variable="As", values=[70.0, 85.1, 93.0], analysis=analysis
    )
    model = ra.ddo.CostBenefitModel(
        benefit_rate=1.2e4,
        interest_rate=0.02,
        service_life=100,
        construction_cost=lambda As: 5000 * As,
        failure_cost=lambda As: 5000 * As + 12 * 1.8e6 + 3e4,
    )
    algorithm = ra.Lqi.from_country(
        "CH",
        indexed=True,
        expected_fatalities_given_failure=12,
        marginal_safety_cost=5_000,
    )
    ddo = ra.Ddo(
        study=study,
        objective=model,
        criterion=algorithm,
    )

    results = ddo.run()
    unconstrained_best = ddo.economic_optimum()
    feasible_best = ddo.optimize()

    assert ddo.criterion.name == "lqi"
    assert ddo.results.equals(results)
    assert list(results.columns) == [
        "As",
        "pf",
        "beta",
        "objective",
        "annualized_safety_cost",
        "expected_fatalities_given_failure",
        "swtp_consequence",
        "target_pf",
        "target_beta",
        "lqi_acceptable",
        "screening_margin",
    ]
    assert np.isfinite(results["beta"]).all()
    assert results.loc[results["As"] == 70.0, "lqi_acceptable"].item() is False
    assert results.loc[results["As"] == 85.1, "lqi_acceptable"].item() is False
    assert results.loc[results["As"] == 93.0, "lqi_acceptable"].item() is True
    assert unconstrained_best["As"] == pytest.approx(85.1)
    assert feasible_best["As"] == pytest.approx(93.0)
    assert ddo.optimize()["As"] == pytest.approx(93.0)


def test_ddo_direct_construction_caches_results():
    study = ra.ddo.DesignStudy(
        variable="As",
        values=[85.1],
        analysis=lambda area: {"pf": 2.5708544377963726e-5},
    )
    criterion = ra.Lqi.from_country(
        "CH",
        indexed=True,
        expected_fatalities_given_failure=12,
        marginal_safety_cost=5_000,
    )

    ddo = ra.Ddo(study=study, criterion=criterion)

    assert isinstance(ddo.criterion, ra.Lqi)
    assert ddo.results is None

    results = ddo.run()
    assert ddo.results is results
    assert ddo.results.loc[0, "target_pf"].item() == pytest.approx(1e-5)
    assert results.loc[0, "lqi_acceptable"].item() is False


def test_ddo_accepts_a_risk_study():
    def scenarios(strengthening):
        scale = 1.0 - strengthening
        return pd.DataFrame(
            {
                "probability": np.array([2.0e-3, 1.5e-3, 2.0e-4]) * scale,
                "fatalities": [0.2, 0.3, 2.0],
                "economic_loss": [1.0e6, 1.5e6, 8.0e6],
            }
        )

    study = ra.ddo.RiskStudy(
        variable="strengthening",
        values=[0.0, 0.5],
        model=ra.ddo.ScenarioRiskModel(scenarios),
    )
    criterion = ra.Lqi.from_swtp(
        5_000_000,
        expected_fatalities_given_failure=12,
        marginal_safety_cost=5_000,
    )

    # The same DDO orchestration accepts a RiskStudy via the pf alias.
    results = ra.Ddo(study=study, criterion=criterion).run()
    assert "pf" in results
    assert "lqi_acceptable" in results
    assert (results["pf"] == results["annual_failure_rate"]).all()


def test_plot_summary_returns_axes_for_design_table():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")

    data = pd.DataFrame(
        {
            "As": [70.0, 85.1, 93.0],
            "pf": [1e-3, 2.5e-5, 5e-6],
            "objective": [10_000.0, 72_000.0, 52_000.0],
        }
    )

    fig, axes = ra.ddo.plot_summary(
        data,
        design="As",
        quantities=["objective", "pf"],
        labels={"As": "cross section", "pf": "failure probability"},
        reference_designs={"optimum": 85.1, "acceptable": 93.0},
        target_failure_probability=1e-4,
    )

    assert len(axes) == 2
    assert axes[1].get_yscale() == "log"
    assert axes[-1].get_xlabel() == "cross section"

    plt.close(fig)
