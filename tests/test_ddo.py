import numpy as np
import pandas as pd
import pytest

import pystra as ra


def test_ddo_namespace_is_explicit():
    assert hasattr(ra, "ddo")
    assert hasattr(ra, "DDO")
    assert hasattr(ra, "DDOCriterion")
    assert hasattr(ra, "LQI")
    assert hasattr(ra, "SWTP")
    assert not hasattr(ra, "RiskStudy")
    assert not hasattr(ra, "plot_summary")


def test_swtp_from_lqi_and_country_lookup():
    swtp = ra.ddo.SWTP.from_lqi(
        gross_domestic_product_per_capita=25010,
        mortality_rate=0.16,
        demographic_constant=16,
        currency="PPPUSD",
        price_year=1999,
    )

    assert swtp.value_per_life == pytest.approx(2_501_000)
    assert ra.ddo.get_swtp("DE") == pytest.approx(1_900_000)
    assert ra.ddo.get_swtp("UK") == pytest.approx(1_700_000)
    assert ra.ddo.get_swtp("US", indexed=True) == pytest.approx(5_220_864, rel=1e-4)
    assert ra.ddo.get_swtp_record("New Zealand").code == "NZ"
    assert ra.ddo.swtp_table().loc["NL", "value_per_life"] == pytest.approx(2_800_000)
    indexed = ra.ddo.swtp_table(indexed=True)
    assert indexed.loc["CH", "price_year"] == 2024
    assert indexed.loc["CH", "index_factor"] == pytest.approx(2.7775, rel=1e-4)


def test_lqi_target_reliability_table_and_ratio():
    k1 = ra.ddo.lqi_k1(
        safety_cost_rate=1_000,
        swtp=ra.ddo.SWTP(5_000_000),
        expected_fatalities=2,
    )

    target = ra.ddo.lqi_target_reliability(k1)
    high_variability = ra.ddo.lqi_target_reliability(k1, variability="high")

    assert k1 == pytest.approx(1e-4)
    assert target.cost_class == "medium"
    assert target.pf == pytest.approx(1e-4)
    assert target.beta == pytest.approx(3.7)
    assert high_variability.pf == pytest.approx(5e-4)
    assert high_variability.beta < target.beta


def test_lqi_builds_target_from_country():
    criterion = ra.LQI.from_country(
        "CH",
        indexed=True,
        expected_fatalities=12,
        marginal_safety_cost=5_000,
    )

    assert criterion.swtp.price_year == 2024
    assert criterion.expected_fatalities == pytest.approx(12)
    assert criterion.k1 == pytest.approx(5_000 / (criterion.swtp.value_per_life * 12))
    assert criterion.target.cost_class == "small"
    assert criterion.target.pf == pytest.approx(1e-5)
    assert criterion.target.beta == pytest.approx(4.2)
    assert criterion.risk_cost(1000, 1e-4) == pytest.approx(
        1000 + criterion.swtp.for_lives(12) * 1e-4
    )


def test_lqi_country_requires_explicit_index_choice():
    with pytest.raises(ValueError, match="indexed=True or indexed=False"):
        ra.LQI.from_country(
            "CH",
            expected_fatalities=12,
            marginal_safety_cost=5_000,
        )


def test_lqi_can_use_explicit_consequence():
    criterion = ra.LQI.from_swtp(
        ra.SWTP(5_000_000),
        consequence=ra.FatalityConsequence(
            people_exposed=20,
            probability_death_given_failure=0.5,
        ),
        marginal_safety_cost=5_000,
    )

    assert criterion.expected_fatalities == pytest.approx(10)
    assert criterion.k1 == pytest.approx(1e-4)


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
        expected_fatalities=10,
    )
    margin_high = ra.ddo.jcss_lqi_acceptability(
        safety_cost=safety_cost,
        failure_rate=failure_probability,
        design=4.4,
        swtp=5e6,
        expected_fatalities=10,
    )

    assert margin_low < 0
    assert margin_high > 0
    assert ra.ddo.jcss_lqi_risk_cost(1000, 1e-4, ra.ddo.SWTP(5e6), 10) == pytest.approx(
        6000
    )
    assert ra.ddo.jcss_systematic_reconstruction_objective(
        benefit_rate=20_000,
        safety_cost=1_000_000,
        failure_rate=1e-4,
        failure_consequence=10_000_000,
        discount_rate=0.02,
    ) == pytest.approx(-55_000)


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
    swtp = ra.ddo.SWTP(5.0e6)

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

    results = study.evaluate(swtp=ra.ddo.SWTP(5.0e6))

    assert list(results.columns) == [
        "strengthening",
        "annual_failure_rate",
        "beta",
        "expected_fatalities",
        "expected_economic_loss",
        "life_safety_cost",
        "total_risk_cost",
        "design",
    ]
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
    algorithm = ra.LQI.from_country(
        "CH",
        indexed=True,
        expected_fatalities=12,
        marginal_safety_cost=5_000,
    )
    ddo = ra.DDO(
        study=study,
        objective=model,
        criterion=algorithm,
    )

    results = ddo.run()
    best = ddo.maximize_objective()

    assert ddo.criterion.name == "lqi"
    assert ddo.getResults().equals(results)
    assert list(results.columns) == [
        "As",
        "pf",
        "beta",
        "objective",
        "annualized_safety_cost",
        "expected_fatalities",
        "swtp_consequence",
        "target_pf",
        "target_beta",
        "lqi_acceptable",
        "lqi_margin",
    ]
    assert np.isfinite(results["beta"]).all()
    assert results.loc[results["As"] == 70.0, "lqi_acceptable"].item() is False
    assert results.loc[results["As"] == 85.1, "lqi_acceptable"].item() is False
    assert results.loc[results["As"] == 93.0, "lqi_acceptable"].item() is True
    assert best["As"] == pytest.approx(85.1)


def test_ddo_lqi_constructor_builds_algorithm():
    study = ra.ddo.DesignStudy(
        variable="As",
        values=[85.1],
        analysis=lambda area: {"pf": 2.5708544377963726e-5},
    )

    ddo = ra.DDO.lqi(
        study=study,
        country="CH",
        indexed=True,
        expected_fatalities=12,
        marginal_safety_cost=5_000,
    )

    assert isinstance(ddo.criterion, ra.LQI)
    with pytest.raises(ValueError, match="not been run"):
        ddo.getResults()

    results = ddo.run()
    assert ddo.getResults().loc[0, "target_pf"].item() == pytest.approx(1e-5)
    assert results.loc[0, "lqi_acceptable"].item() is False


def test_ddo_is_keyword_only_and_validates_lqi_conflicts():
    study = ra.ddo.DesignStudy(
        variable="As",
        values=[85.1],
        analysis=lambda area: {"pf": 2.5708544377963726e-5},
    )
    criterion = ra.LQI.from_swtp(
        5_000_000,
        expected_fatalities=12,
        marginal_safety_cost=5_000,
    )

    with pytest.raises(TypeError):
        ra.DDO(study, criterion)

    with pytest.raises(ValueError, match="criterion cannot be combined"):
        ra.DDO.lqi(
            study=study,
            criterion=criterion,
            country="CH",
            indexed=True,
        )


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
