"""Published tutorial cases, checked independently of their analytical helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gumbel_r, norm

import pystra as ra
from pystra.active_learning import (
    ActiveLearning,
    AllCriteria,
    BetaBounds,
    BetaStability,
    PCESurrogate,
)

spec = importlib.util.spec_from_file_location(
    "literature_benchmarks",
    Path(__file__).resolve().parents[1]
    / "docs/source/notebooks/literature_benchmarks.py",
)
benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmarks)


def stiffness_deflection(parameters):
    """Independent planar FE assembly from the paper's Figure 4, in SI units."""
    elasticity_h, elasticity_d, area_h, area_d, *loads = parameters
    nodes = np.vstack(
        (
            np.column_stack((np.arange(0, 25, 4), np.zeros(7))),
            np.column_stack((np.arange(2, 23, 4), np.full(6, 2))),
        )
    )
    horizontal = [(i, i + 1) for i in range(6)] + [(i, i + 1) for i in range(7, 12)]
    diagonal = [(i, i + 7) for i in range(6)] + [(i + 1, i + 7) for i in range(6)]
    stiffness = np.zeros((26, 26))
    for members, rigidity in (
        (horizontal, elasticity_h * area_h),
        (diagonal, elasticity_d * area_d),
    ):
        for start, end in members:
            direction = nodes[end] - nodes[start]
            length = np.linalg.norm(direction)
            projection = np.r_[-direction, direction] / length
            dofs = [2 * start, 2 * start + 1, 2 * end, 2 * end + 1]
            stiffness[np.ix_(dofs, dofs)] += (
                rigidity / length * np.outer(projection, projection)
            )
    force = np.zeros(26)
    force[15::2] = -np.asarray(loads)
    free = np.setdiff1d(np.arange(26), [0, 1, 13])
    displacement = np.zeros(26)
    displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], force[free])
    return -displacement[7]  # bottom node at x=12 m, vertical degree of freedom


def test_truss_unit_load_expression_against_independent_fe():
    nominal = np.array([2.1e11, 2.1e11, 2e-3, 1e-3, *([5e4] * 6)])
    rng = np.random.default_rng(812)
    cases = nominal * np.exp(rng.normal(0, 0.3, size=(20, 10)))
    individual = np.tile(nominal, (6, 1))
    individual[:, 4:] = np.eye(6) * 5e4
    cases = np.vstack((cases, individual))
    np.testing.assert_allclose(
        benchmarks.truss_deflection(*cases.T),
        [stiffness_deflection(case) for case in cases],
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        benchmarks.truss_deflection(*cases.T),
        benchmarks.truss_deflection(
            *np.column_stack((cases[:, :4], cases[:, 4:][:, ::-1])).T
        ),
        rtol=2e-15,
    )


def test_truss_reference_against_direct_simulation_and_published_mc():
    reference = benchmarks.truss_reference(power=18)
    probability = np.mean(reference)
    integration_error = np.std(reference, ddof=1) / np.sqrt(len(reference))
    published = 1.52e-3
    published_error = np.sqrt(published * (1 - published) / 1e6)
    assert abs(probability - published) < 4 * (integration_error + published_error)
    # Fresh direct simulation of all ten physical variables: no product
    # reduction, conditional integration, PySTRA transforms or surrogate.
    rng = np.random.default_rng(981)
    n_samples = 1_000_000
    variance = np.log1p(0.1**2)
    properties = rng.lognormal(
        np.log([2.1e11, 2.1e11, 2e-3, 1e-3]) - variance / 2,
        np.sqrt(variance),
        size=(n_samples, 4),
    )
    scale = 7.5e3 * np.sqrt(6) / np.pi
    loads = rng.gumbel(5e4 - np.euler_gamma * scale, scale, size=(n_samples, 6))
    observed = np.mean(benchmarks.truss_limit_state(*properties.T, *loads.T) <= 0)
    assert abs(observed - probability) < 4 * (
        np.sqrt(probability * (1 - probability) / n_samples) + integration_error
    )


def test_truss_form_and_sorm_published_comparison():
    model = benchmarks.truss_model()
    state = ra.LimitState(benchmarks.truss_limit_state)
    form = ra.FORM(model, state)
    result = form.run()
    assert result.converged
    assert result.failure_probability == pytest.approx(0.76e-3, abs=0.005e-3)
    for fit_type in ("cf", "pf"):
        fit = {"cf": "curve", "pf": "point"}[fit_type]
        sorm = ra.SORM(model, state, form=form, options=ra.SORMOptions(fit=fit))
        sorm.run()
        assert sorm._results_valid
        # Breitung accuracy against the probability reference.
        assert sorm._pf2_breitung == pytest.approx(1.53e-3, rel=0.06)
        if fit_type == "cf":
            # Modified Breitung matches Table 3 at its printed precision;
            # the paper does not identify which SORM formula it used.
            assert sorm._pf2_breitung_m == pytest.approx(1.63e-3, abs=0.005e-3)


def test_hat_quadrature_against_importance_sampling_and_exact_cubic():
    probability = benchmarks.hat_reference()
    rng = np.random.default_rng(913)
    # Shift only the sum coordinate towards failure. This integral samples
    # the original 2-D event, independently of the conditional quadrature.
    points = rng.normal(size=(400_000, 2))
    points[:, 0] += 3
    first = 0.25 + (points[:, 0] + points[:, 1]) / np.sqrt(2)
    second = 0.25 + (points[:, 0] - points[:, 1]) / np.sqrt(2)
    weighted = (benchmarks.hat_limit_state(first, second) <= 0) * np.exp(
        -3 * points[:, 0] + 4.5
    )
    assert abs(weighted.mean() - probability) < 4 * weighted.std(ddof=1) / np.sqrt(
        len(weighted)
    )
    training = rng.normal(size=(60, 2))
    surrogate = PCESurrogate(degree=3, method="ols", seed=4)
    surrogate.fit(training, benchmarks.hat_limit_state(*(training + 0.25).T))
    query = rng.uniform(-5, 5, size=(2000, 2))
    np.testing.assert_allclose(
        surrogate.predict(query)[0],
        benchmarks.hat_limit_state(*(query + 0.25).T),
        rtol=1e-11,
        atol=1e-9,
    )


@pytest.mark.parametrize("seed", [7, 23, 101])
@pytest.mark.parametrize("problem", ["truss", "hat"])
def test_literature_active_pce(problem, seed):
    model = getattr(benchmarks, f"{problem}_model")()
    function = getattr(benchmarks, f"{problem}_limit_state")
    reference = (
        benchmarks.truss_reference(power=18).mean()
        if problem == "truss"
        else benchmarks.hat_reference()
    )
    analysis = ActiveLearning(
        model=model,
        limit_state=ra.LimitState(function),
        surrogate=PCESurrogate(
            degree=(1, 2, 3, 4, 5),
            q_norm=0.75 if problem == "truss" else 1,
            max_interaction=2,
            seed=seed,
        ),
        n_initial=60 if problem == "truss" else 30,
        n_candidates=12_000,
        n_estimation=400_000,
        max_iterations=180,
        stopping_criterion=AllCriteria(
            criteria=(BetaBounds(consecutive=2), BetaStability(consecutive=2))
        ),
        rng=seed,
    )
    result = analysis.run()
    assert result.converged, result.status
    assert result.n_limit_state_evaluations < 240
    assert (
        abs(result.failure_probability - reference)
        < 4 * np.sqrt(reference * (1 - reference) / result.n_estimation)
        + 0.05 * reference
    )
    # Check cancellation of false positives and negatives cannot hide a poor fit.
    points = np.random.default_rng(892).normal(size=(200_000, model.n_marg))
    if problem == "truss":
        variance = np.log1p(0.1**2)
        properties = np.exp(
            np.log([2.1e11, 2.1e11, 2e-3, 1e-3])
            - variance / 2
            + np.sqrt(variance) * points[:, :4]
        )
        scale = 7.5e3 * np.sqrt(6) / np.pi
        loads = gumbel_r.ppf(
            norm.cdf(points[:, 4:]), loc=5e4 - np.euler_gamma * scale, scale=scale
        )
        truth = function(*properties.T, *loads.T) <= 0
    else:
        truth = function(*(points + 0.25).T) <= 0
    prediction = analysis.surrogate_model.predict(points)[0] <= 0
    assert np.mean(truth != prediction) < 0.1 * reference
