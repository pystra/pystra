"""Every analysis returns an immutable record of its results."""

import dataclasses
import pickle

import numpy as np
import pytest

import pystra as ra


def normal_model(first="R", second="S"):
    model = ra.StochasticModel()
    model.add_variable(ra.Normal(first, 10.0, 1.0))
    model.add_variable(ra.Normal(second, 5.0, 1.0))
    return model


def margin(offset=0.0):
    return ra.LimitState(lambda R, S: R - S - offset)


def options(samples=None, target_cov=None):
    result = ra.AnalysisOptions()
    if samples is not None:
        result.set_samples(samples)
    if target_cov is not None:
        result.target_cov = target_cov
    return result


def form():
    analysis = ra.FORM(stochastic_model=normal_model(), limit_state=margin())
    return analysis, analysis.run()


def test_form_result_copies_the_solution_into_read_only_arrays():
    analysis, result = form()
    assert (result.method, result.status, result.converged) == (
        "FORM",
        "converged",
        True,
    )
    assert result.beta == result.design_index == pytest.approx(5 / np.sqrt(2))
    assert result.failure_probability == analysis.get_failure()
    np.testing.assert_array_equal(
        result.design_point_x, analysis.get_design_point(uspace=False)
    )
    np.testing.assert_array_equal(result.design_point_u, analysis.get_design_point())
    np.testing.assert_array_equal(result.alpha, analysis.get_alpha())
    assert result.variable_names == ("R", "S")
    assert result.n_limit_state_evaluations == analysis.get_no_function_calls() > 0
    for vector in (result.design_point_x, result.design_point_u, result.alpha):
        assert not vector.flags.writeable
        with pytest.raises(ValueError):
            vector[0] = 0.0
    analysis.u[0] = 99.0
    assert result.design_point_u[0] != 99.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.beta = 0.0


def test_results_compare_by_value_and_form_tabulates_by_variable():
    _, first = form()
    _, second = form()
    assert first == second and first is not second
    with pytest.raises(TypeError):
        hash(first)
    table = first.to_dataframe()
    assert list(table.index) == ["R", "S"] and table.index.name == "variable"
    assert list(table.columns) == ["design_point_x", "design_point_u", "alpha"]
    np.testing.assert_array_equal(table["design_point_x"], first.design_point_x)
    text = first.summary()
    assert text.startswith("FORM result")
    assert "converged" in text and "Limit-state evaluations" in text


def test_unconverged_form_result_has_no_estimate_or_table():
    analysis = ra.FORM(stochastic_model=normal_model(), limit_state=margin())
    analysis.options.set_imax(1)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = analysis.run()
    assert result.status == "not_converged" and not result.converged
    assert result.failure_probability is result.design_point_x is result.alpha is None
    assert "unavailable" in result.summary()
    with pytest.raises(ValueError, match="no design point"):
        result.to_dataframe()


def test_status_must_be_a_documented_value():
    _, result = form()
    with pytest.raises(ValueError, match="Unknown status"):
        dataclasses.replace(result, status="done")


@pytest.mark.parametrize(
    "fit, fit_type, shape", [("curve", "cf", (1,)), ("point", "pf", (2, 1))]
)
def test_sorm_result_reports_breitung_and_each_approximation(fit, fit_type, shape):
    model = ra.StochasticModel()
    model.add_variable(ra.Lognormal("R", 10.0, 1.5))
    model.add_variable(ra.Normal("S", 5.0, 1.0))
    first = ra.FORM(stochastic_model=model, limit_state=margin())
    form_result = first.run()
    sorm = ra.SORM(stochastic_model=model, limit_state=margin(), form=first)
    result = sorm.run(fit_type=fit_type)
    assert isinstance(result, ra.SORMResult) and result.status == "converged"
    assert (result.fit, result.formula) == (fit, "breitung")
    assert result.failure_probability == result.approximations["breitung"]
    assert result.failure_probability == sorm.pf2_breitung
    assert result.beta == sorm.betag_breitung
    assert result.approximations["modified_breitung"] == sorm.pf2_breitung_m
    assert result.form == form_result
    expected = sorm.kappa if fit == "curve" else sorm.kappa_pf
    np.testing.assert_array_equal(result.curvatures, expected)
    assert result.curvatures.shape == shape
    assert result.n_limit_state_evaluations > 0
    with pytest.raises(TypeError):
        result.approximations["breitung"] = 0.0


def curved_sorm(c):
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X1", 0.0, 1.0))
    model.add_variable(ra.Normal("X2", 0.0, 1.0))
    limit_state = ra.LimitState(lambda X1, X2: 3 - X2 - c * X1**2)
    analysis = ra.FORM(stochastic_model=model, limit_state=limit_state)
    analysis.run()
    return ra.SORM(stochastic_model=model, limit_state=limit_state, form=analysis).run()


def test_sorm_without_a_defined_breitung_estimate_is_not_converged(capsys):
    # FORM stops at (0, 3), where the surface curves towards the origin more
    # sharply than Breitung's formula allows: kappa = -2c < -1/beta.
    result = curved_sorm(0.2)
    assert result.status == "not_converged" and not result.converged
    assert result.failure_probability is None and result.beta is None
    assert dict(result.approximations) == {"breitung": None, "modified_breitung": None}
    assert result.curvatures[0] == pytest.approx(-0.4, abs=1e-5)


def test_sorm_keeps_breitung_when_only_the_modified_formula_is_undefined(capsys):
    result = curved_sorm(0.16)
    assert result.status == "converged"
    assert result.approximations["modified_breitung"] is None
    assert result.failure_probability == result.approximations["breitung"] > 0


def crude(samples, offset=3.0, target_cov=None, seed=11):
    np.random.seed(seed)
    analysis = ra.CrudeMonteCarlo(
        analysis_options=options(samples, target_cov),
        limit_state=margin(offset),
        stochastic_model=normal_model(),
    )
    return analysis, analysis.run()


def test_crude_monte_carlo_result_reports_precision_against_its_target():
    analysis, result = crude(2000)
    assert (
        isinstance(result, ra.SimulationResult) and result.method == "CrudeMonteCarlo"
    )
    assert result.failure_probability == analysis.get_failure() > 0
    assert result.beta == analysis.get_beta()
    assert result.coefficient_of_variation == analysis.cov_q_bar[analysis.k - 1] > 0.05
    assert result.n_samples == result.n_limit_state_evaluations == 2000
    assert result.status == "precision_not_met" and not result.converged
    assert result.failure_probability is not None
    _, early = crude(2000, target_cov=0.2)
    assert early.status == "completed" and early.converged
    assert early.n_samples == 1000 and early.coefficient_of_variation <= 0.2


def test_simulation_without_failures_has_infinite_coefficient_of_variation():
    _, result = crude(1000, offset=-100.0)
    assert result.failure_probability == 0 and result.beta == np.inf
    assert result.coefficient_of_variation == np.inf
    assert result.status == "precision_not_met"


def test_importance_and_line_sampling_record_their_form_point():
    np.random.seed(5)
    importance = ra.ImportanceSampling(
        analysis_options=options(2000),
        limit_state=margin(),
        stochastic_model=normal_model(),
    )
    result = importance.run()
    assert result.method == "ImportanceSampling"
    assert isinstance(result.diagnostics["form"], ra.FORMResult)
    np.testing.assert_array_equal(
        result.diagnostics["form"].design_point_u, importance.point.ravel()
    )
    lines = ra.LineSampling(
        analysis_options=options(50),
        limit_state=margin(),
        stochastic_model=normal_model(),
    )
    result = lines.run()
    assert (result.method, result.status) == ("LineSampling", "completed")
    assert result.failure_probability == lines.Pf
    assert result.coefficient_of_variation == lines.cov and result.n_samples == 50
    np.testing.assert_array_equal(result.diagnostics["direction"], lines.alpha)
    assert not result.diagnostics["direction"].flags.writeable
    assert result.diagnostics["form"].converged


def test_subset_simulation_records_its_levels():
    np.random.seed(7)
    analysis = ra.SubsetSimulation(
        analysis_options=options(500),
        limit_state=margin(),
        stochastic_model=normal_model(),
    )
    result = analysis.run()
    levels = result.diagnostics
    assert (result.method, result.status) == ("SubsetSimulation", "completed")
    assert levels["n_levels"] == analysis.n_levels > 1
    assert levels["samples_per_level"] == 500
    assert result.n_samples == 500 * analysis.n_levels
    np.testing.assert_array_equal(levels["thresholds"], analysis.thresholds)
    np.testing.assert_array_equal(
        levels["conditional_probabilities"], analysis.conditional_probs
    )
    assert result.failure_probability == pytest.approx(
        np.prod(levels["conditional_probabilities"])
    )


def test_simulation_records_survive_pickling():
    _, result = crude(1000)
    assert pickle.loads(pickle.dumps(result)) == result


def test_distribution_analysis_returns_its_samples():
    np.random.seed(9)
    analysis = ra.DistributionAnalysis(
        analysis_options=options(300),
        limit_state=margin(),
        stochastic_model=normal_model(),
    )
    result = analysis.run()
    assert isinstance(result, ra.DistributionAnalysisResult)
    assert result.status == "completed" and result.n_samples == 300
    assert result.samples_x.shape == (300, 2) and result.limit_state_values.shape == (
        300,
    )
    np.testing.assert_array_equal(
        result.limit_state_values, result.samples_x[:, 0] - result.samples_x[:, 1]
    )
    assert not hasattr(result, "failure_probability")


def test_system_form_result_holds_each_component_record():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0.0, 1.0))
    model.add_variable(ra.Normal("Y", 0.0, 1.0))
    system = ra.SeriesSystem(
        [ra.Component("x", lambda X: 3 - X), ra.Component("y", lambda Y: 3 - Y)]
    )
    analysis = ra.SystemFORM(system, model)
    result = analysis.run()
    assert isinstance(result, ra.SystemFORMResult) and result.status == "converged"
    assert result.failure_probability == analysis.get_failure()
    assert list(result.component_results) == ["x", "y"]
    assert all(isinstance(r, ra.FORMResult) for r in result.component_results.values())
    lower, upper = result.bounds
    assert lower <= result.failure_probability <= upper
    np.testing.assert_array_equal(result.correlation, analysis.correlation)
    assert not result.correlation.flags.writeable
    assert result.n_limit_state_evaluations == sum(
        r.n_limit_state_evaluations for r in result.component_results.values()
    )


@pytest.mark.parametrize("numerical", [True, False])
def test_sensitivity_result_holds_read_only_derivatives(numerical):
    analysis = ra.SensitivityAnalysis(margin(), normal_model())
    result = analysis.run(numerical=numerical)
    assert isinstance(result, ra.SensitivityResult) and result.status == "converged"
    assert result.approach == ("numerical" if numerical else "closed_form")
    assert result.form.beta == result.beta == pytest.approx(5 / np.sqrt(2))
    assert result.marginal["R"]["mean"] == pytest.approx(1 / np.sqrt(2), rel=1e-2)
    with pytest.raises(TypeError):
        result.marginal["R"]["mean"] = 0.0
    assert (result.correlation is None) == numerical
    assert (result.delta is None) != numerical
    table = result.to_dataframe()
    assert list(table.columns) == ["Variable", "Parameter", "∂β/∂θ"] and len(table) == 4
    assert result.n_limit_state_evaluations >= result.form.n_limit_state_evaluations


def test_strong_maximum_result_is_a_diagnostic_record():
    analysis, _ = form()
    test = ra.StrongMaximumTest(analysis, point_number=200, seed=3)
    result = test.run()
    assert isinstance(result, ra.StrongMaximumResult) and result.status == "completed"
    assert (
        result.has_competing_points is False and "not a certificate" in result.message
    )
    assert result.point_number == 200 and result.points_u.shape == (200, 2)
    np.testing.assert_array_equal(
        result.points("near_failure"), test.get_points("near_failure")
    )
    np.testing.assert_array_equal(
        result.points("far_safe", space="x"), test.get_points("far_safe", uspace=False)
    )
    assert result.n_limit_state_evaluations == test.evaluation_count == 202
    assert not hasattr(result, "failure_probability")
    assert "Strong" in result.summary() or "StrongMaximumTest" in result.summary()
