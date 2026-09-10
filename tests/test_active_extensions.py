"""Independent mathematical checks for PC-Kriging, FBR and active IS."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

import pystra as ra
from pystra.active_learning._pce import _hermite_basis
from pystra.active_learning import (
    ActiveLearning,
    AllCriteria,
    BetaBounds,
    BetaStability,
    BootstrapBounds,
    EnsembleSurrogate,
    FBRLearning,
    ImportanceSamplingEstimator,
    LearningStep,
    LearningThreshold,
    PCKrigingSurrogate,
    PCESurrogate,
)


def test_pc_kriging_against_executed_uqlab_prediction():
    data = json.loads((Path(__file__).parent / "data/uqlab_active.json").read_text())
    for case in data["cases"]:
        surrogate = PCKrigingSurrogate(
            degree=case["degree"],
            correlation="gaussian",
            length_scale=case["length_scale"],
            optimize=False,
            noise=case["noise"],
        )
        surrogate.fit(case["points"], case["values"])
        expected = case["expected"]
        mean, std = surrogate.predict(case["query"])
        np.testing.assert_array_equal(
            surrogate.fit_result.trend.indices, case["indices"]
        )
        np.testing.assert_allclose(mean, expected["mean"], atol=2e-9, rtol=2e-8)
        np.testing.assert_allclose(std**2, expected["variance"], atol=2e-9, rtol=2e-8)
        np.testing.assert_allclose(
            surrogate.fit_result.coefficients,
            expected["coefficients"],
            atol=2e-9,
            rtol=2e-8,
        )
        assert surrogate.fit_result.process_variance == pytest.approx(
            expected["process_variance"], rel=2e-8
        )
    decision = FBRLearning().select_replicates(data["fbr"]["predictions"])
    assert decision.index == np.argmin(data["fbr"]["expected"])
    assert decision.score == min(data["fbr"]["expected"])


def test_pc_kriging_variance_against_augmented_system():
    rng = np.random.default_rng(88)
    points = rng.normal(size=(18, 2))
    values = 2 + points[:, 0] + 0.3 * np.sin(2 * points[:, 1])
    surrogate = PCKrigingSurrogate(
        degree=1, length_scale=[0.7, 1.3], optimize=False, correlation="gaussian"
    )
    surrogate.fit(points, values)
    fit = surrogate.fit_result
    powers = np.asarray(fit.trend.indices)
    basis = _hermite_basis(points, powers)
    difference = (points[:, None, :] - points[None, :, :]) / fit.length_scale
    correlation = np.exp(-np.sum(difference**2, axis=2) / 2) + 1e-8 * np.eye(
        len(points)
    )
    # Solve the constrained minimum-variance predictor directly, without
    # GLS whitening or the implementation's trend correction expression.
    system = np.block(
        [[correlation, basis], [basis.T, np.zeros((basis.shape[1], basis.shape[1]))]]
    )
    query = rng.uniform(-3, 3, size=(20, 2))
    mean, std = surrogate.predict(query)
    simple_variances = []
    for i, point in enumerate(query):
        cross = np.exp(-np.sum(((points - point) / fit.length_scale) ** 2, axis=1) / 2)
        trend = _hermite_basis(point[None, :], powers)[0]
        weights = np.linalg.solve(system, np.r_[cross, trend])[: len(points)]
        assert mean[i] == pytest.approx(weights @ values, abs=2e-10)
        variance = fit.process_variance * (
            1 - 2 * weights @ cross + weights @ correlation @ weights
        )
        assert std[i] ** 2 == pytest.approx(variance, abs=2e-10)
        simple_variances.append(
            fit.process_variance * (1 - cross @ np.linalg.solve(correlation, cross))
        )
    assert np.max(std**2 - simple_variances) > 1e-3  # trend uncertainty matters
    with pytest.raises(FrozenInstanceError):
        fit.process_variance = 0


def test_pc_kriging_exactness_scaling_and_fit_reset(monkeypatch):
    rng = np.random.default_rng(99)
    points = rng.normal(size=(40, 2))
    values = 3 + 2 * points[:, 0] + points[:, 1] ** 2
    query = rng.normal(size=(20, 2))
    surrogate = PCKrigingSurrogate(degree=2)
    surrogate.fit(points, values)
    mean, std = surrogate.predict(query)
    np.testing.assert_allclose(mean, 3 + 2 * query[:, 0] + query[:, 1] ** 2, atol=1e-11)
    assert not std.any()
    with pytest.raises(ValueError):
        surrogate.fit(points, values[:-1])
    with pytest.raises(RuntimeError):
        surrogate.predict(query)
    for multiplier in (1, -1000):
        surrogate.fit(points, multiplier * (values + 0.3 * np.sin(points[:, 0] * 3)))
        prediction = surrogate.predict(query)
        if multiplier == 1:
            baseline = prediction
        else:
            # Fixed optimizer path is invariant under output sign/scaling up
            # to optimizer tolerance; compare with a generous numerical bound.
            np.testing.assert_allclose(
                prediction[0] / multiplier, baseline[0], atol=3e-3
            )
            np.testing.assert_allclose(
                prediction[1] / abs(multiplier), baseline[1], atol=3e-3
            )
    import pystra.active_learning.pc_kriging as module

    monkeypatch.setattr(
        module,
        "minimize",
        lambda *args, **kwargs: type("Failed", (), {"success": False})(),
    )
    with pytest.raises(RuntimeError, match="optimization"):
        surrogate.fit(points, values + np.sin(points[:, 0]))
    assert surrogate.fit_result is None


@pytest.mark.parametrize(
    "settings",
    [
        dict(noise=0),
        dict(length_scale=[]),
        dict(length_scale=-1),
        dict(n_restarts=-1),
        dict(optimize="yes"),
        dict(length_scale=1000),
    ],
)
def test_pc_kriging_invalid_settings(settings):
    with pytest.raises(ValueError):
        PCKrigingSurrogate(**settings)


def test_fbr_requires_replicates_and_handles_zero_and_equal_moments():
    ensemble = np.array(
        [[-1, -1, -1, 3], [-np.sqrt(3), -np.sqrt(3), np.sqrt(3), np.sqrt(3)]]
    )
    np.testing.assert_allclose(ensemble.mean(axis=1), 0)
    np.testing.assert_allclose(ensemble.std(axis=1), np.sqrt(3))
    assert FBRLearning().select_replicates(ensemble).index == 1
    # Complementary votes must tie exactly, including an odd replicate count.
    assert FBRLearning().select_replicates([[-1, 1, 1], [-1, -1, 1]]).index == 0
    assert FBRLearning().select_replicates([[0, 0, 0, 0]]).threshold_satisfied
    assert FBRLearning().select_replicates([[0, 0, 1, 1]]).score == 0
    with pytest.raises(TypeError, match="replicate"):
        FBRLearning().select([0, 0], [1, 1])
    for values in ([], [[1]], [[np.nan, 0]], [1, 2]):
        with pytest.raises(ValueError):
            FBRLearning().select_replicates(values)


def test_bootstrap_bounds_window_endpoints_and_composition():
    step = LearningStep(
        0.01,
        0,
        30,
        (0.009, 0.011),
        (2, 3),
        False,
        bootstrap_probability_band=(0.0096, 0.0104),
    )
    rule = BootstrapBounds()
    assert not rule.should_stop((step,))
    assert rule.should_stop((step, step))
    for bad in (
        replace(step, bootstrap_probability_band=None),
        replace(step, failure_probability=0),
        replace(step, estimation_converged=False),
        replace(step, bootstrap_probability_band=(0.009, 0.011)),
    ):
        assert not rule.should_stop((step, bad))
    assert AllCriteria(criteria=(BetaBounds(), rule)).requires_bootstrap
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    limit_state = ra.LimitState(lambda X: 3 - X)
    with pytest.raises(TypeError, match="EnsembleSurrogate"):
        ActiveLearning(
            model, limit_state, surrogate="pc_kriging", learning_function="fbr"
        )
    with pytest.raises(ValueError, match="fixed IID"):
        ActiveLearning(
            model,
            limit_state,
            surrogate="pce",
            learning_function="fbr",
            estimator=ImportanceSamplingEstimator(centers=[[3]]),
        )


def test_bootstrap_predictions_are_consistent_across_batches():
    rng = np.random.default_rng(71)
    points = rng.normal(size=(35, 2))
    surrogate = PCESurrogate(degree=2, seed=4)
    assert isinstance(surrogate, EnsembleSurrogate)
    with pytest.raises(RuntimeError):
        surrogate.predict_replicates(points)
    surrogate.fit(points, 2 + points[:, 0] + 0.1 * np.sin(points[:, 1]))
    replicates = surrogate.predict_replicates(points)
    np.testing.assert_allclose(
        replicates,
        np.vstack(
            [
                surrogate.predict_replicates(points[:10]),
                surrogate.predict_replicates(points[10:]),
            ]
        ),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        replicates.std(axis=1, ddof=1), surrogate.predict(points)[1]
    )
    replicates[:] = np.nan
    assert np.isfinite(surrogate.predict_replicates(points)).all()


@pytest.mark.parametrize("dimension", [2, 20])
def test_importance_sampling_ensemble_error_and_variance(dimension):
    reference = norm.sf(5)
    center = np.full((1, dimension), 5 / np.sqrt(dimension))
    estimator = ImportanceSamplingEstimator(centers=center, n_samples=4000)
    estimates, errors = [], []
    for seed in range(30):
        result = estimator.estimate(
            lambda u: (5 - u.sum(axis=1) / np.sqrt(dimension), np.zeros(len(u))),
            dimension=dimension,
            rng=np.random.default_rng(seed),
        )
        assert result.converged
        assert result.sampling_interval is None
        estimates.append(result.failure_probability)
        errors.append(result.failure_probability * result.sampling_cov)
        assert result.diagnostics[0].max_weight <= 10 * (1 + 1e-12)
    observed = np.std(estimates, ddof=1)
    assert abs(np.mean(estimates) - reference) < 4 * observed / np.sqrt(len(estimates))
    assert 0.6 < np.sqrt(np.mean(np.square(errors))) / observed < 1.6


def test_importance_weighted_bands_two_modes_and_pool_cap():
    estimator = ImportanceSamplingEstimator(centers=[[-4, 0], [4, 0]], n_samples=50000)

    def predict(u):
        return 4 - np.abs(u[:, 0]), np.full(len(u), 0.1)

    result = estimator.explore(
        predict, dimension=2, rng=np.random.default_rng(40), n_candidates=300
    )
    assert len(result.points) == 300
    assert np.mean(result.mean <= 0) > 100 * result.estimate.failure_probability
    for value, reference in zip(
        (
            *result.probability_band[:1],
            result.estimate.failure_probability,
            result.probability_band[1],
        ),
        2 * norm.sf([4.2, 4, 3.8]),
    ):
        assert value == pytest.approx(reference, rel=0.06)
    larger = estimator.explore(
        predict, dimension=2, rng=np.random.default_rng(40), n_candidates=600
    )
    assert larger.estimate == result.estimate
    assert larger.probability_band == result.probability_band
    with pytest.raises(ValueError):
        result.points[0, 0] = 0


def test_importance_endpoints_invalid_probability_and_insufficient_samples():
    estimator = ImportanceSamplingEstimator(centers=[[4]], n_samples=100)
    for sign, expected in [(1, "no_failures"), (-1, "all_failures")]:
        result = estimator.estimate(
            lambda u: (np.full(len(u), sign), np.zeros(len(u))),
            dimension=1,
            rng=np.random.default_rng(4),
        )
        assert not result.converged and np.isinf(result.sampling_cov)
        assert result.status in (expected, "invalid_probability")
    rare = ImportanceSamplingEstimator(centers=[[0]], n_samples=100)
    result = rare.estimate(
        lambda u: (2 - u[:, 0], np.zeros(len(u))),
        dimension=1,
        rng=np.random.default_rng(5),
    )
    assert not result.converged
    with pytest.raises(ValueError, match="dimension"):
        estimator.estimate(
            lambda u: (u[:, 0], u[:, 0]), dimension=2, rng=np.random.default_rng(7)
        )
    # Force an out-of-range finite IS estimate: retained and explicitly invalid.
    estimator._sample = lambda dimension, rng: (np.zeros((100, 1)), np.full(100, 2.0))
    result = estimator.estimate(
        lambda u: (-np.ones(100), np.zeros(100)),
        dimension=1,
        rng=np.random.default_rng(7),
    )
    assert result.failure_probability == 1 and result.status == "invalid_probability"
    assert not result.converged and result.diagnostics[0].raw_probability == 2


@pytest.mark.parametrize(
    "settings",
    [
        dict(centers=[]),
        dict(centers=[[np.nan]]),
        dict(centers=[[1]], scale=0),
        dict(centers=[[1]], defensive_fraction=1),
        dict(centers=[[1]], n_samples=1),
        dict(centers=[[1]], min_effective_failures=1),
    ],
)
def test_importance_invalid_settings(settings):
    with pytest.raises(ValueError):
        ImportanceSamplingEstimator(**settings)


def problem(name):
    model = ra.StochasticModel()
    for i in range(2):
        model.add_variable(ra.Normal(f"x{i}", 0, 1))
    if name == "linear":
        function = lambda x0, x1: 4.5 - (x0 + x1) / np.sqrt(2)
        reference = norm.sf(4.5)
        centers = [[4.5 / np.sqrt(2)] * 2]
    elif name == "two_tail":
        function = lambda x0, x1: 16 - x0**2
        reference = 2 * norm.sf(4)
        centers = [[-4, 0], [4, 0]]
    else:
        function = lambda x0, x1: np.minimum(
            3 + 0.1 * (x0 - x1) ** 2 - abs(x0 + x1) / np.sqrt(2),
            3 * np.sqrt(2) - abs(x0 - x1),
        )
        reference = 0.00445733149063
        centers = None
    return model, function, reference, centers


@pytest.mark.parametrize("seed", [7, 23, 101])
@pytest.mark.parametrize("name", ["linear", "two_tail"])
def test_active_importance_benchmarks(name, seed):
    model, function, reference, centers = problem(name)
    analysis = ActiveLearning(
        model=model,
        limit_state=ra.LimitState(function),
        surrogate="kriging",
        surrogate_kwargs={"noise": 1e-8, "n_restarts": 0},
        estimator=ImportanceSamplingEstimator(centers=centers),
        stopping_criterion=AllCriteria(
            criteria=(BetaBounds(consecutive=2), BetaStability(consecutive=2))
        ),
        rng=seed,
    )
    pytest.importorskip("sklearn")
    result = analysis.run()
    assert result.converged, result.status
    assert result.n_evaluations < 160
    assert (
        abs(result.failure_probability - reference)
        < 4 * reference * result.sampling_cov + 0.05 * reference
    )
    # Independent proposal sample and likelihood weights validate classifications.
    points, weights = ImportanceSamplingEstimator(
        centers=centers, n_samples=100000
    )._sample(2, np.random.default_rng(911))
    truth = function(*points.T) <= 0
    prediction = analysis.surrogate_model.predict(points)[0] <= 0
    assert np.mean(weights * (truth != prediction)) < 0.1 * reference


@pytest.mark.parametrize("seed", [7, 23, 101])
def test_four_branch_pc_kriging(seed):
    model, function, reference, _ = problem("four_branch")
    analysis = ActiveLearning(
        model=model,
        limit_state=ra.LimitState(function),
        surrogate="pc_kriging",
        surrogate_kwargs={"degree": (1, 2, 3)},
        n_initial=50,
        n_candidates=12000,
        n_estimation=100000,
        stopping_criterion=AllCriteria(
            criteria=(
                LearningThreshold(),
                BetaBounds(consecutive=2),
                BetaStability(consecutive=2),
            )
        ),
        learning_threshold=3,
        max_iterations=200,
        rng=seed,
    )
    result = analysis.run()
    assert result.converged and result.n_evaluations < 251
    assert (
        abs(result.failure_probability - reference)
        < 4 * np.sqrt(reference * (1 - reference) / result.n_estimation)
        + 0.05 * reference
    )
    points = np.random.default_rng(892).normal(size=(100000, 2))
    truth = function(*points.T) <= 0
    prediction = analysis.surrogate_model.predict(points)[0] <= 0
    assert np.mean(truth != prediction) < 0.1 * reference
    along = (points[:, 0] + points[:, 1]) / np.sqrt(2)
    across = (points[:, 0] - points[:, 1]) / np.sqrt(2)
    for region in (
        along > 3 + 0.2 * across**2,
        along < -3 - 0.2 * across**2,
        across > 3,
        across < -3,
    ):
        assert region.sum() > 50
        assert np.mean(~prediction[region]) < 0.15


@pytest.mark.parametrize("seed", [7, 23, 101])
def test_fbr_lognormal_beam(seed):
    model = ra.StochasticModel()
    means = np.array([0.15, 0.3, 5, 30000, 0.01])
    stds = means * np.array([0.05, 0.05, 0.01, 0.15, 0.2])
    for name, mean, std in zip(("b", "h", "length", "elasticity", "load"), means, stds):
        model.add_variable(ra.Lognormal(name, mean, std))
    function = lambda b, h, length, elasticity, load: 0.015 - 5 * load * length**4 / (
        32 * elasticity * b * h**3
    )
    powers = np.array([-1, -3, 4, -1, 1])
    variance = np.log1p((stds / means) ** 2)
    reference = norm.sf(
        (np.log(0.015) - np.log(5 / 32) - powers @ (np.log(means) - variance / 2))
        / np.sqrt(powers**2 @ variance)
    )
    analysis = ActiveLearning(
        model=model,
        limit_state=ra.LimitState(function),
        surrogate="pce",
        surrogate_kwargs={"q_norm": 0.75, "max_interaction": 2},
        learning_function="fbr",
        n_candidates=20000,
        n_estimation=100000,
        rng=seed,
    )
    result = analysis.run()
    assert result.converged and result.n_evaluations < 100
    assert all(step.bootstrap_probability_band is not None for step in result.history)
    assert (
        abs(result.failure_probability - reference)
        < 4 * np.sqrt(reference * (1 - reference) / result.n_estimation)
        + 0.05 * reference
    )
    points = np.random.default_rng(892).normal(size=(100000, 5))
    truth = analysis._evaluate(points) <= 0
    prediction = analysis.surrogate_model.predict(points)[0] <= 0
    assert np.mean(truth != prediction) < 0.1 * reference


@pytest.mark.parametrize("correlation", ["gaussian", "matern52"])
def test_pc_kriging_likelihood_gradient(correlation):
    from scipy.optimize import approx_fprime

    rng = np.random.default_rng(43)
    points = rng.normal(size=(20, 2))
    values = 2 + points[:, 0] + np.sin(2 * points[:, 1])
    basis = np.column_stack((np.ones(20), points))
    surrogate = PCKrigingSurrogate(correlation=correlation)
    scales = np.array([-0.3, 0.2])
    _, gradient = surrogate._objective(scales, points, basis, values)
    numerical = approx_fprime(
        scales, lambda z: surrogate._objective(z, points, basis, values)[0], 1e-6
    )
    np.testing.assert_allclose(gradient, numerical, rtol=3e-4, atol=5e-5)


def test_bootstrap_unanimity_cannot_certify_an_unobserved_failure_mode():
    # Within the initial region this two-mode model is exactly linear.
    axis = np.linspace(-1, 1, 5)
    x0, x1 = np.meshgrid(axis, axis)
    points = np.column_stack((x0.ravel(), x1.ravel()))
    response = np.minimum(3 - points[:, 0], 50 - 10 * points[:, 1] ** 2)
    surrogate = PCESurrogate(degree=1, seed=7)
    surrogate.fit(points, response)
    query = np.random.default_rng(44).normal(size=(20000, 2))
    replicates = surrogate.predict_replicates(query)
    probabilities = np.mean(replicates <= 0, axis=0)
    assert probabilities.min() == probabilities.max()
    probability = np.mean(surrogate.predict(query)[0] <= 0)
    step = LearningStep(
        probability,
        1,
        25,
        (probability, probability),
        (3, 3),
        True,
        bootstrap_probability_band=(probabilities.min(), probabilities.max()),
    )
    assert BootstrapBounds().should_stop((step, step))
    true_probability = 1 - (1 - norm.sf(3)) * (1 - 2 * norm.sf(np.sqrt(5)))
    assert probability < true_probability / 10  # independent validation is essential


def test_fbr_four_branch_reports_budget_exhaustion():
    model, function, _, _ = problem("four_branch")
    analysis = ActiveLearning(
        model=model,
        limit_state=ra.LimitState(function),
        surrogate="pce",
        learning_function="fbr",
        rng=7,
        n_initial=30,
        max_iterations=3,
        stopping_criterion=BootstrapBounds(tolerance=0.001),
    )
    with pytest.warns(RuntimeWarning, match="max_iterations"):
        result = analysis.run()
    assert not result.converged and result.status == "max_iterations"
