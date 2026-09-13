"""Analytic scenarios, evaluator substitution and failure retention in studies."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from scipy.stats import norm

import pystra as ra
from pystra.assessment import (
    AssessmentCase,
    ReliabilityEstimate,
    analyze_case,
    assess_cases,
    evaluate_reliability,
)
from pystra.reporting import reliability_row
from tests.test_calibration_workflows import generic_model


def normal_case(name="reference", resistance=10):
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("R", resistance, 1))
    model.add_variable(ra.Normal("S", 5, 1))
    return AssessmentCase(
        name,
        model,
        ra.LimitState(lambda R, S: R - S),
        metadata={
            "units": "kN",
            "reference_period_years": 50,
            "dependence": "independent",
        },
    )


def linear_adapter(model, limit_state, *, options=None):
    # Independent closed-form reliability for this adapter's linear R/S model.
    z = model.constants.get("thickness", 1)
    r, s = model.variable("R"), model.variable("S")
    beta = (z * r.mean - s.mean) / np.hypot(z * r.std, s.std)
    return ReliabilityEstimate(method="linear external", beta=beta)


def factor_problem():
    cases = ra.LoadCombination(
        cases={"traffic": [ra.Normal("R", 10, 1), ra.Normal("S", 5, 1)]},
        constants=[ra.Constant("thickness", 1)],
        roles=ra.VariableRoles(resistance=("R",), variable=("S",)),
        leading_actions={"traffic": ("S",)},
        limit_state=lambda thickness, R, S: thickness * R - S,
    )
    return ra.calibration.FactorCalibrationProblem(
        cases,
        nominal_values={"R": 10, "S": 5},
        design_parameter="thickness",
    )


def test_assessment_default_and_external_match_analytic_scenarios():
    cases = [normal_case(), normal_case("corroded", 9)]
    form = assess_cases(cases, target_beta=3)
    external = assess_cases(cases, evaluator=linear_adapter, target_beta=3)
    assert form.converged and external.converged
    for i, expected in enumerate([5 / np.sqrt(2), 4 / np.sqrt(2)]):
        assert form.cases[i].reliability.beta == pytest.approx(expected)
        assert external.cases[i].reliability.beta == pytest.approx(expected)
        assert external.cases[i].target_margin == pytest.approx(expected - 3)
    assert list(external.to_frame().case_name) == ["reference", "corroded"]
    assert external.to_frame().iloc[0].metadata["reference_period_years"] == 50


def test_assessment_snapshots_inputs_results_and_engineering_metadata():
    case = normal_case()
    metadata = case.metadata
    metadata["units"] = "changed"
    model = case.model
    model.variable("R").name = "changed"
    seen = []

    def mutating_adapter(model, limit_state, *, options=None):
        result = linear_adapter(model, limit_state)
        model.variable("R").name = "mutated by solver"
        seen.append(model)
        return result

    first = assess_cases([case], evaluator=mutating_adapter)
    second = assess_cases([case], evaluator=mutating_adapter)
    table = first.to_frame()
    table.loc[0, "beta"] = 999
    table.at[0, "metadata"]["units"] = "changed in table"
    assert first.cases[0].reliability == second.cases[0].reliability
    assert first.cases[0].case.metadata["units"] == "kN"
    assert case.model.names == ["R", "S"]
    assert seen[0] is not seen[1]


def test_assessment_retains_form_and_external_failures_without_target_margins():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = assess_cases(
            [normal_case()], options=ra.FORMOptions(max_iterations=1), target_beta=3
        )
    assert not result.converged
    assert result.cases[0].target_margin is None
    assert result.cases[0].reliability.method == "FORM"
    assert np.isnan(result.to_frame().beta.iloc[0])

    def failing_adapter(model, limit_state, *, options=None):
        raise ra.AnalysisError("external mesh failed")

    result = assess_cases(
        [normal_case(), normal_case("second")], evaluator=failing_adapter, target_beta=3
    )
    assert len(result.cases) == 2 and not result.converged
    assert result.to_frame().message.tolist() == ["external mesh failed"] * 2
    assert result.to_frame().target_margin.isna().all()


def test_resultless_failure_labels_identify_wrapped_and_returned_solvers():
    class ExternalSolver:
        def __init__(self, model, limit_state, *, options=None):
            pass

        def run(self):
            raise ra.AnalysisError("external mesh failed")

    callbacks = (
        partial(ExternalSolver),
        lambda model, limit_state, options=None: ExternalSolver(model, limit_state),
    )
    for callback in callbacks:
        result = assess_cases([normal_case("bridge")], evaluator=callback)
        row = result.to_frame().iloc[0]
        assert row.case_name == "bridge"
        assert row.method == "ExternalSolver"
        assert row.message == "external mesh failed"
        assert not row.converged


def test_anonymous_resultless_failure_has_a_readable_label_and_original_message():
    def fail():
        raise ra.AnalysisError("external mesh failed")

    anonymous = lambda model, limit_state, options=None: fail()
    for callback in (anonymous, partial(anonymous)):
        result = assess_cases([normal_case("bridge")], evaluator=callback)
        row = result.to_frame().iloc[0]
        assert row.case_name == "bridge"
        assert row.method == "external evaluator (AnalysisError)"
        assert row.message == "external mesh failed"
        assert not row.converged


def test_sampling_method_constructor_is_accepted_and_retains_precision_status():
    case = normal_case(resistance=6)
    # One fixed block avoids a random stopping-time comparison; check against
    # the binomial standard error of the independent analytic probability.
    options = ra.SimulationOptions(n_samples=20000, block_size=20000, target_cov=0.1)
    result = assess_cases(
        [case], options=options, evaluator=partial(ra.CrudeMonteCarlo, rng=431)
    )
    record = result.cases[0].reliability
    expected = norm.sf(1 / np.sqrt(2))
    assert record.failure_probability == pytest.approx(
        expected, abs=5 * np.sqrt(expected * (1 - expected) / 20000)
    )
    assert record.converged
    limited = assess_cases(
        [case],
        options=replace(options, target_cov=1e-6),
        evaluator=partial(ra.CrudeMonteCarlo, rng=431),
    )
    assert not limited.converged
    assert (
        limited.cases[0].reliability.failure_probability == record.failure_probability
    )
    assert limited.to_frame().status.iloc[0] == "precision_not_met"
    assert np.isnan(limited.to_frame().failure_probability.iloc[0])


def test_estimate_only_adapter_solves_and_verifies_but_cannot_derive_factors():
    problem = factor_problem()
    solutions = ra.calibration.solve_designs(
        problem, target_beta=3, bracket=(0.5, 2), evaluator=linear_adapter
    )
    assert solutions.converged
    z = solutions.designs[0].design_value
    assert (10 * z - 5) / np.hypot(z, 1) == pytest.approx(3, abs=1e-4)
    verified = ra.calibration.verify_designs(
        problem, z, evaluator=linear_adapter, target_beta=3
    )
    assert abs(verified[0].target_margin) <= 1e-4
    assert analyze_case(problem.cases, evaluator=linear_adapter).beta == pytest.approx(
        5 / np.sqrt(2)
    )
    with pytest.raises(ValueError, match="normal standard space"):
        ra.calibration.derive_factors(solutions, method="coeff")
    with pytest.raises(ValueError, match="normal standard space"):
        ra.calibration.solve_designs(
            problem, target_beta=3, method="alpha", evaluator=linear_adapter
        )


def test_alternative_form_factory_has_projection_capability():
    problem = factor_problem()
    result = ra.calibration.solve_designs(
        problem, target_beta=3, method="alpha", evaluator=ra.FORM
    )
    assert result.converged
    factors = ra.calibration.derive_factors(result, method="coeff")
    selected = ra.calibration.select_factors(factors)
    designs = ra.calibration.design_with_factors(problem, selected)
    checked = ra.calibration.verify_designs(problem, designs, target_beta=3)
    assert checked[0].reliability.converged
    assert abs(checked[0].target_margin) < 1e-4


def test_target_solve_does_not_reuse_success_after_external_failure():
    calls = 0

    def later_failure(model, limit_state, *, options=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return linear_adapter(model, limit_state)
        raise ra.AnalysisError("solver stopped at new candidate")

    solutions = ra.calibration.solve_designs(
        factor_problem(), target_beta=3, evaluator=later_failure
    )
    assert not solutions.converged
    outcome = solutions.designs[0]
    assert not outcome.reliability.converged
    assert outcome.reliability.beta is None
    assert outcome.residual is None
    assert "new candidate" in outcome.message
    table = solutions.reliability_frame()
    assert not table.solve_converged.iloc[0]
    assert np.isnan(table.beta.iloc[0])


def test_normalized_study_accepts_estimates_and_retains_failed_grid_points():
    outcomes = iter(
        [
            ReliabilityEstimate(method="external", beta=3),
            ReliabilityEstimate(
                method="external", beta=8, status="precision_not_met", message="budget"
            ),
        ]
    )
    study = ra.calibration.CodeCalibration(
        live_load_ratios=[0, 1], dead_load_ratios=[0.5]
    )
    result = study.run(
        generic_model(),
        ra.calibration.CodeFactors(1, 1, 1, 1),
        target_beta=2,
        evaluator=lambda model, limit_state, options=None: next(outcomes),
    )
    assert result.beta[0, 0] == 3 and np.isnan(result.beta[0, 1])
    assert result.cases[0].target_margin == 1
    assert result.cases[1].reliability.beta == 8
    assert result.cases[1].target_margin is None
    assert result.to_frame().status.tolist() == ["completed", "precision_not_met"]


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_successful_external_probabilities_are_rejected(value):
    with pytest.raises(ValueError, match="failure_probability"):
        ReliabilityEstimate(method="external", failure_probability=value)


def test_tail_index_is_preserved_and_untrusted_estimates_are_masked():
    result = ReliabilityEstimate(method="external", failure_probability=0, beta=40)
    assert reliability_row(result)["beta"] == 40
    row = reliability_row(
        replace(result, status="not_converged", message="iteration limit")
    )
    assert np.isnan(row["beta"]) and np.isnan(row["failure_probability"])
    assert row["message"] == "iteration limit"


def test_invalid_specifications_and_programming_errors_are_not_mislabeled_as_failures():
    case = normal_case()
    for error in (ra.ModelError("bad input"), TypeError("adapter bug")):

        def invalid(model, limit_state, options=None):
            raise error

        with pytest.raises(type(error), match=str(error)):
            assess_cases([case], evaluator=invalid)
    with pytest.raises(ValueError, match="unique"):
        assess_cases([case, case])
    with pytest.raises(TypeError, match="FORMOptions"):
        evaluate_reliability(
            case.model, case.limit_state, options=ra.SimulationOptions()
        )
