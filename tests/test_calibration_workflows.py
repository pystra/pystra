"""Analytic cases and regressions for the v2 calibration contracts."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

import pystra as ra
from tests.test_calibration import setup1, setup2


def generic_model(*, uncertain_errors=False):
    error = ra.Lognormal if uncertain_errors else None
    return ra.calibration.NormalizedReliabilityModel(
        resistance=(
            ra.Lognormal("capacity", 1, 0.08)
            if uncertain_errors
            else ra.Normal("capacity", 1, 0.1)
        ),
        dead_load=ra.Normal("self_weight", 1, 0.08),
        permanent_load=ra.Normal("superimposed", 1, 0.1),
        live_load=ra.Normal("traffic", 1, 0.1),
        resistance_error=(
            error("resistance_error", 1, 0.05)
            if error
            else ra.Constant("resistance_error", 1)
        ),
        load_error=(
            error("load_error", 1, 0.1) if error else ra.Constant("load_error", 1)
        ),
        nominal_values=ra.calibration.NominalValues(1, 1, 1, 1),
    )


def problem(fixture=setup1):
    cases, nominals, target = fixture()
    return (
        ra.calibration.FactorCalibrationProblem(
            cases, nominal_values=nominals, design_parameter="z"
        ),
        target,
    )


def test_form_result_survives_rerun_and_failed_result_has_no_estimate():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("R", 10, 1))
    model.add_variable(ra.Normal("S", 5, 1))
    analysis = ra.FORM(model, ra.LimitState(lambda R, S: R - S))
    first = analysis.run()
    assert first.beta == pytest.approx(5 / np.sqrt(2))
    analysis.options.set_imax(1)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        failed = analysis.run()
    assert not failed.converged
    assert failed.beta is failed.failure_probability is failed.design_point is None
    assert first.beta == pytest.approx(5 / np.sqrt(2))
    assert first.design_point == pytest.approx((7.5, 7.5))


def test_normalized_design_grid_matches_linear_normal_solution_and_endpoints():
    model = generic_model()
    factors = ra.calibration.CodeFactors(0.8, 1.2, 1.5, 1.6)
    study = ra.calibration.CodeCalibration(
        live_load_ratios=[0, 0.5, 1], dead_load_ratios=[0, 1]
    )
    result = study.run(model, factors, target_beta=3)
    assert result.converged
    for case in result.cases:
        aq, ag, z = case.live_load_ratio, case.dead_load_ratio, case.design_value
        expected_z = ((1 - aq) * (ag * 1.2 + (1 - ag) * 1.5) + aq * 1.6) / 0.8
        variance = (
            (z * 0.1) ** 2
            + ((1 - aq) * ag * 0.08) ** 2
            + ((1 - aq) * (1 - ag) * 0.1) ** 2
            + (aq * 0.1) ** 2
        )
        beta = (z - 1) / np.sqrt(variance)
        assert z == pytest.approx(expected_z)
        assert case.reliability.beta == pytest.approx(beta, abs=1e-6)
        assert case.reliability.failure_probability == pytest.approx(norm.sf(beta))
        assert case.target_margin == pytest.approx(beta - 3, abs=1e-6)
    assert result.beta.shape == (2, 3)


def test_factor_comparison_is_fresh_and_snapshots_do_not_alias_models_or_tables():
    model = generic_model(uncertain_errors=True)
    factors = ra.calibration.CodeFactors(0.8, 1.2, 1.5, 1.6)
    grid = np.array([0.5])
    study = ra.calibration.CodeCalibration(live_load_ratios=grid, dead_load_ratios=grid)
    first = study.run(model, factors)
    second = study.run(model, replace(factors, phi=1.0))
    assert first.beta.item() == pytest.approx(4.1234292054, abs=1e-6)
    assert second.beta.item() == pytest.approx(2.6148664228, abs=1e-6)
    grid[0] = 0
    model.resistance.name = "modified"
    copy = first.model
    copy.resistance.name = "also_modified"
    table = first.to_frame()
    table.loc[0, "beta"] = 999
    beta = first.beta
    beta[0, 0] = 999
    assert first.model.resistance.name == "capacity"
    assert first.live_load_ratios == (0.5,)
    assert first.beta.item() == pytest.approx(4.1234292054, abs=1e-6)


@pytest.mark.parametrize("ratios", [[], [-0.1], [1.1], [np.nan], [[0.5]]])
def test_invalid_ratio_grids_are_rejected(ratios):
    with pytest.raises(ValueError):
        ra.calibration.CodeCalibration(live_load_ratios=ratios, dead_load_ratios=[0.5])


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_nominals_and_factors_are_rejected(value):
    with pytest.raises(ValueError):
        ra.calibration.CodeFactors(value, 1, 1, 1)
    with pytest.raises(ValueError):
        ra.calibration.NominalValues(value, 1, 1, 1)


def test_nonconverged_generic_cases_are_retained_and_not_plotted_as_envelopes():
    options = ra.AnalysisOptions()
    options.set_imax(1)
    study = ra.calibration.CodeCalibration(
        live_load_ratios=[0.3, 0.7], dead_load_ratios=[0.5]
    )
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = study.run(
            generic_model(uncertain_errors=True),
            ra.calibration.CodeFactors(0.8, 1.2, 1.5, 1.6),
            options=options,
            target_beta=4,
        )
    assert len(result.cases) == 2
    assert not result.converged
    assert np.all(np.isnan(result.beta))
    assert all(c.target_margin is None for c in result.cases)
    with pytest.raises(ValueError, match="fully converged"):
        ra.calibration.plot_calibration({"failed": result})


def test_generic_correlated_model_uses_explicit_dependence():
    model = generic_model()
    rho = np.eye(4)
    rho[0, 3] = rho[3, 0] = 0.4
    model = replace(model, copula=ra.GaussianCopula(rho))
    result = ra.calibration.CodeCalibration(
        live_load_ratios=[1], dead_load_ratios=[0]
    ).run(model, ra.calibration.CodeFactors(1, 1, 1, 1.5))
    expected = 0.5 / np.sqrt((1.5 * 0.1) ** 2 + 0.1**2 - 2 * 1.5 * 0.4 * 0.1 * 0.1)
    assert result.beta.item() == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("method", ["coeff", "matrix"])
@pytest.mark.parametrize("fixture", [setup1, setup2])
def test_factor_results_are_invariant_to_case_and_variable_order(method, fixture):
    original, target = problem(fixture)
    cases = original.cases
    baseline = ra.calibration.derive_factors(
        ra.calibration.solve_designs(original, target_beta=target), method=method
    )
    roles = ra.VariableRoles(
        tuple(reversed(cases.roles.resistance)),
        tuple(reversed(cases.roles.other)),
        tuple(reversed(cases.roles.variable)),
    )
    reordered = ra.LoadCombination(
        cases=dict(reversed(list(cases.cases.items()))),
        limit_state=cases.limit_state,
        constants=cases.constants,
        roles=roles,
        leading_actions=dict(reversed(list(cases.leading_actions.items()))),
    )
    alternative = ra.calibration.FactorCalibrationProblem(
        reordered, nominal_values=original.nominal_values, design_parameter="z"
    )
    actual = ra.calibration.derive_factors(
        ra.calibration.solve_designs(alternative, target_beta=target), method=method
    )
    for kind in ("resistance", "loads", "combinations"):
        expected = baseline.to_frame(kind)
        aligned = actual.to_frame(kind).reindex(
            index=expected.index, columns=expected.columns
        )
        assert np.allclose(aligned, expected, atol=1e-6)
    selected = ra.calibration.select_factors(
        actual, resistance="minimum", loads="maximum", combinations="maximum"
    )
    assert selected.governing
    for i, lead in enumerate(selected.leading_actions):
        assert all(
            selected.combinations[i][selected.load_names.index(n)] == 1 for n in lead
        )


@pytest.mark.parametrize("method", ["root", "alpha"])
def test_design_parameter_can_be_renamed_and_start_at_target(method):
    original, target = problem()
    cases = original.cases
    constants = cases.constants
    constants.pop("z")
    constants["scale"] = ra.Constant("scale", 1)
    renamed = ra.LoadCombination(
        cases=cases.cases,
        limit_state=lambda scale, **values: cases.limit_state(z=scale, **values),
        constants=constants,
        roles=cases.roles,
        leading_actions=cases.leading_actions,
    )
    new = ra.calibration.FactorCalibrationProblem(
        renamed, nominal_values=original.nominal_values, design_parameter="scale"
    )
    solved = ra.calibration.solve_designs(new, target_beta=target, method=method)
    assert solved.converged
    assert "scale" in solved.to_frame() and "z" not in solved.to_frame()
    factors = ra.calibration.derive_factors(solved)
    assert ra.calibration.design_with_factors(new, factors).values[0] > 0
    repeated = ra.calibration.solve_designs(
        new,
        target_beta=target,
        method=method,
        initial_value=solved.designs[0].design_value,
    )
    assert repeated.converged
    assert renamed.constants["scale"].get_value() == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evaluations": 1},
        {"bracket": (1, 2)},
        {"method": "alpha", "max_evaluations": 1},
    ],
)
def test_unmet_target_is_explicit_and_cannot_produce_factors(kwargs):
    inputs, target = problem()
    result = ra.calibration.solve_designs(inputs, target_beta=target, **kwargs)
    assert not result.converged
    assert len(result.designs) == 2
    assert any(not d.converged and d.message for d in result.designs)
    with pytest.raises(ValueError, match="failed target"):
        ra.calibration.derive_factors(result)


def test_inner_failure_cannot_become_a_calibrated_design():
    inputs, target = problem()
    options = ra.AnalysisOptions()
    options.set_imax(1)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = ra.calibration.solve_designs(
            inputs, target_beta=target, options=options
        )
    assert not result.converged
    assert all(
        d.residual is None and not d.reliability.converged for d in result.designs
    )


def test_final_verification_is_included_in_target_solve_budget():
    inputs, target = problem()
    completed = ra.calibration.solve_designs(inputs, target_beta=target)
    budget = completed.designs[0].evaluations - 1
    limited = ra.calibration.solve_designs(
        inputs, target_beta=target, max_evaluations=budget
    )
    first = limited.designs[0]
    assert first.evaluations == budget
    assert not first.converged
    assert "budget exhausted" in first.message


def test_normal_form_result_keeps_finite_index_when_probability_underflows():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    result = ra.FORM(model, ra.LimitState(lambda X: 40 - X)).run()
    assert result.converged
    assert result.beta == pytest.approx(40)
    assert result.failure_probability == 0


def test_explicit_cases_require_roles_for_factors_and_reject_unknown_overrides():
    inputs, _ = problem()
    cases = inputs.cases
    unclassified = ra.LoadCombination(
        cases=cases.cases, constants=cases.constants, limit_state=cases.limit_state
    )
    with pytest.raises(ValueError, match="roles"):
        ra.calibration.FactorCalibrationProblem(
            unclassified, nominal_values=inputs.nominal_values, design_parameter="z"
        )
    with pytest.raises(ValueError, match="Unknown overrides"):
        cases.stochastic_model(overrides={"typo": ra.Constant("typo", 2)})
    with pytest.raises(ValueError, match="exactly"):
        ra.calibration.FactorCalibrationProblem(
            cases, nominal_values={"R": 1}, design_parameter="z"
        )


def test_bracketed_target_solve_and_full_design_verification():
    inputs, target = problem()
    result = ra.calibration.solve_designs(
        inputs, target_beta=target, bracket=(2, 4), tolerance=1e-6
    )
    assert result.converged
    factors = ra.calibration.select_factors(
        ra.calibration.derive_factors(result),
        resistance="minimum",
        loads="maximum",
        combinations="maximum",
    )
    designs = ra.calibration.design_with_factors(inputs, factors)
    checks = ra.calibration.verify_designs(
        inputs, max(designs.values), target_beta=target
    )
    assert len(checks) == 2
    assert all(c.reliability.converged and c.target_margin >= -1e-4 for c in checks)
    assert designs.governing_cases


def test_plot_uses_smallest_dead_load_ratio_and_does_not_execute_a_study():
    import matplotlib.pyplot as plt

    result = ra.calibration.CodeCalibration(
        live_load_ratios=[0.2, 0.8], dead_load_ratios=[1, 0]
    ).run(generic_model(), ra.calibration.CodeFactors(0.8, 1.2, 1.5, 1.6))
    fig, ax = ra.calibration.plot_calibration({"example": result})
    assert np.allclose(ax.lines[0].get_ydata(), result.beta[1])
    plt.close(fig)
