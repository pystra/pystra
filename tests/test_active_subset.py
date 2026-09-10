"""Subset sampling invariants, analytic rare events and active reliability."""

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

import pystra as ra
from pystra.active_learning import (
    ActiveLearning,
    AllCriteria,
    BetaBounds,
    BetaStability,
    EnrichmentEstimator,
    EnrichmentResult,
    LearningThreshold,
    ReliabilityEstimate,
    SubsetSimulationEstimator,
)
from pystra.active_learning.subset import _indicator_variance
from .test_active_learning import ExactLinear, benchmark


def linear_predictor(points):
    return 5 - points.sum(axis=1) / np.sqrt(points.shape[1]), np.zeros(len(points))


def test_conditional_kernel_preserves_truncated_normal_and_other_coordinates():
    # Begin with independently drawn exact stationary seeds; this detects a
    # missing normal-density acceptance correction in a random-walk kernel.
    rng = np.random.default_rng(814)
    seeds = rng.normal(size=(2000, 2))
    seeds[:, 0] = norm.isf(rng.uniform(0, norm.sf(2), len(seeds)))
    predict = lambda points: (2 - points[:, 0], np.zeros(len(points)))
    sampler = SubsetSimulationEstimator(n_samples=20_000)
    mean, std = predict(seeds)
    points, values, _, chains, acceptance, calls = sampler._conditional(
        predict,
        seeds,
        mean,
        std,
        0,
        0,
        rng,
    )
    assert np.all(points[:, 0] >= 2) and np.all(values <= 0)
    expected = norm.pdf(2) / norm.sf(2)
    assert points[:, 0].mean() == pytest.approx(expected, abs=0.025)
    assert points[:, 0].var() == pytest.approx(
        1 + 2 * expected - expected**2, abs=0.025
    )
    assert abs(points[:, 1].mean()) < 0.06
    assert points[:, 1].var() == pytest.approx(1, abs=0.08)
    assert 0.1 < acceptance < 0.9
    assert calls == 18_000
    assert len(np.unique(chains)) == 2000


def test_chain_variance_detects_repeated_states():
    chains = np.repeat(np.arange(100), 10)
    indicators = np.repeat(np.arange(100) < 10, 10)
    variance, factor = _indicator_variance(indicators, chains)
    iid, _ = _indicator_variance(indicators, None)
    assert factor > 9
    assert variance > 9 * iid


@pytest.mark.parametrize("dimension", [2, 20])
def test_replicated_rare_event_bias_and_reported_variation(dimension):
    reference = norm.sf(5)
    estimator = SubsetSimulationEstimator()
    results = [
        estimator.estimate(
            linear_predictor, dimension=dimension, rng=np.random.default_rng(seed)
        )
        for seed in range(30)
    ]
    probabilities = np.array([result.failure_probability for result in results])
    empirical_std = probabilities.std(ddof=1)
    # An ensemble check separates stochastic fluctuation from systematic bias.
    assert (
        abs(probabilities.mean() - reference)
        < 4 * empirical_std / np.sqrt(len(results)) + 0.02 * reference
    )
    reported_std = np.sqrt(
        np.mean(
            [
                (result.failure_probability * result.sampling_cov) ** 2
                for result in results
            ]
        )
    )
    assert 0.6 * empirical_std < reported_std < 2 * empirical_std
    assert all(result.converged for result in results)
    for result in results:
        assert result.sampling_interval is None
        assert result.sampling_dependence == "dependent"
        replicates = [run.failure_probability for run in result.diagnostics]
        assert result.failure_probability == pytest.approx(np.mean(replicates))
        assert (
            result.sampling_cov
            >= np.std(replicates, ddof=1)
            / np.sqrt(len(replicates))
            / result.failure_probability
            - 1e-14
        )
        assert any(
            level.variance_factor > 1.5
            for run in result.diagnostics
            for level in run.levels
        )
        for run in result.diagnostics:
            assert run.failure_probability == pytest.approx(
                np.prod([level.conditional_probability for level in run.levels])
            )
            assert run.levels[-1].threshold == 0
            assert np.all(np.diff([level.threshold for level in run.levels]) < 0)


def test_exploration_bands_use_nested_probabilities_not_pooled_counts():
    estimator = SubsetSimulationEstimator(n_samples=4000, n_replications=8)
    predict = lambda points: (4 - points[:, 0], np.full(len(points), 0.1))
    exploration = estimator.explore(
        predict, dimension=2, rng=np.random.default_rng(42), n_candidates=10000
    )
    expected = (norm.sf(4.2), norm.sf(4), norm.sf(3.8))
    actual = (
        *exploration.probability_band[:1],
        exploration.estimate.failure_probability,
        exploration.probability_band[1],
    )
    np.testing.assert_allclose(actual, expected, rtol=0.3)
    assert (
        np.mean(exploration.mean <= 0) > 100 * exploration.estimate.failure_probability
    )
    assert not exploration.points.flags.writeable
    with pytest.raises(ValueError):
        exploration.points.setflags(write=True)
    assert len(np.unique(exploration.points, axis=0)) == len(exploration.points)
    assert len(exploration.points) <= 10000


@pytest.mark.parametrize(
    "settings,status", [({"max_levels": 1}, "max_levels"), ({}, "stalled")]
)
def test_incomplete_subset_runs_cannot_claim_sampling_precision(settings, status):
    estimator = SubsetSimulationEstimator(n_samples=100, **settings)
    predict = (
        linear_predictor
        if status == "max_levels"
        else lambda points: (np.ones(len(points)), np.zeros(len(points)))
    )
    result = estimator.estimate(predict, dimension=2, rng=np.random.default_rng(4))
    assert not result.converged
    assert result.status == status
    assert result.failure_probability == 0
    assert np.isinf(result.sampling_cov)
    assert not LearningThreshold().accepts_estimate(result)


def test_zero_boundary_counts_as_failure_without_false_precision():
    result = SubsetSimulationEstimator(n_samples=100).estimate(
        lambda points: (np.zeros(len(points)), np.zeros(len(points))),
        dimension=1,
        rng=np.random.default_rng(1),
    )
    assert result.converged and result.failure_probability == 1
    assert np.isinf(result.sampling_cov)
    assert not LearningThreshold().accepts_estimate(result)


@pytest.mark.parametrize(
    "settings",
    [
        {"n_samples": 10},
        {"conditional_probability": 0},
        {"conditional_probability": 0.9999},
        {"proposal_scale": 0},
        {"proposal_scale": 1.1},
        {"max_levels": 0},
        {"n_replications": 1},
    ],
)
def test_invalid_subset_configuration(settings):
    with pytest.raises(ValueError):
        SubsetSimulationEstimator(**settings)


class RecordedEstimator(EnrichmentEstimator):
    def __init__(self):
        self.draws = []
        self.incomplete = False

    def explore(self, predict, *, dimension, rng, n_candidates):
        points = rng.normal(size=(20, dimension))
        self.draws.append(points.copy())
        mean, std = predict(points)
        estimate = ReliabilityEstimate(
            0.02,
            0.01,
            None,
            None,
            20,
            "test",
            "dependent",
            converged=not self.incomplete,
        )
        return EnrichmentResult(points, mean, std, estimate, (0.02, 0.02))

    def estimate(self, predict, *, dimension, rng):
        points = rng.normal(size=(20, dimension))
        self.draws.append(points.copy())
        predict(points)
        return ReliabilityEstimate(0.02, 0.01, None, None, 20, "test", "dependent")


def test_resampling_after_each_fit_separates_probability_and_selection():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("x", 0, 1))
    estimator, surrogate = RecordedEstimator(), ExactLinear()
    analysis = ActiveLearning(
        model=model,
        limit_state=ra.LimitState(lambda x: 2 - x),
        surrogate=surrogate,
        estimator=estimator,
        stopping_criterion=BetaStability(consecutive=2),
        rng=7,
    )
    state = np.random.get_state()
    result = analysis.run()
    assert result.converged and len(result.history) == 3
    assert len(estimator.draws) == 4
    np.testing.assert_array_equal(estimator.draws[0], estimator.draws[1])
    assert not np.array_equal(estimator.draws[0], estimator.draws[-1])
    assert all(step.failure_probability == 0.02 for step in result.history)
    assert len(np.unique(surrogate.designs[-1], axis=0)) == result.n_evaluations
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    assert result == analysis.run()
    estimator.incomplete = True
    analysis.max_iterations = 3
    with pytest.warns(RuntimeWarning, match="max_iterations"):
        unfinished = analysis.run()
    assert not unfinished.converged and not analysis.results_valid
    assert all(not step.estimation_converged for step in unfinished.history)
    assert not BetaBounds().should_stop(unfinished.history)
    estimator.incomplete = False
    estimator.estimate = lambda predict, **kwargs: ReliabilityEstimate(
        0.02,
        np.inf,
        None,
        None,
        20,
        "test",
        "dependent",
        converged=False,
        status="max_levels",
    )
    with pytest.warns(RuntimeWarning, match="estimation_failed"):
        final_failure = analysis.run()
    assert final_failure.status == "estimation_failed"
    assert not final_failure.converged and not analysis.results_valid
    assert result.converged


def test_subset_rejects_invalid_predictor_outputs():
    estimator = SubsetSimulationEstimator(n_samples=100)
    with pytest.raises(ValueError, match="wrong number"):
        estimator.estimate(
            lambda points: (np.ones(2), np.zeros(2)),
            dimension=1,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match="overflowed"):
        estimator.explore(
            lambda points: (np.ones(len(points)), np.full(len(points), 1e308)),
            dimension=1,
            rng=np.random.default_rng(1),
            n_candidates=10,
        )


def rare_problem(name):
    if name == "four_branch":
        return benchmark(name)
    model = ra.StochasticModel()
    for i in range(2):
        model.add_variable(ra.Normal(f"x{i}", 0, 1))
    if name == "linear":
        function = lambda x0, x1: 4.5 - (x0 + x1) / np.sqrt(2)
        reference = norm.sf(4.5)
    else:
        function = lambda x0, x1: 16 - x0**2
        reference = 2 * norm.sf(4)
    return model, ra.LimitState(function), reference


@pytest.mark.parametrize("seed", [7, 23, 101])
@pytest.mark.parametrize("problem", ["linear", "two_tail", "four_branch"])
def test_active_kriging_subset_benchmarks(problem, seed):
    pytest.importorskip("sklearn")
    model, limit_state, reference = rare_problem(problem)
    analysis = ActiveLearning(
        model=model,
        limit_state=limit_state,
        surrogate_kwargs={"n_restarts": 0, "noise": 1e-8},
        estimator=SubsetSimulationEstimator(),
        stopping_criterion=AllCriteria(
            criteria=(
                BetaBounds(consecutive=2, target_cov=0.25),
                BetaStability(consecutive=2, target_cov=0.25),
            )
        ),
        rng=seed,
        max_iterations=200,
    )
    result = analysis.run()
    assert result.converged
    assert result.n_evaluations < 250
    assert (
        abs(result.failure_probability - reference)
        < 4 * reference * result.sampling_cov + 0.05 * reference
    )
    rng = np.random.default_rng(892)
    points = rng.normal(size=(100_000, 2))
    if problem == "linear":
        points += 4.5 / np.sqrt(2)
        weights = np.exp(-4.5 * points.sum(axis=1) / np.sqrt(2) + 4.5**2 / 2)
    elif problem == "two_tail":
        points[:, 0] += rng.choice([-4, 4], size=len(points))
        weights = np.exp(
            norm.logpdf(points[:, 0])
            - logsumexp(
                np.stack(
                    (norm.logpdf(points[:, 0] - 4), norm.logpdf(points[:, 0] + 4))
                ),
                axis=0,
            )
            + np.log(2)
        )
    else:
        weights = np.ones(len(points))
    truth = analysis._evaluate(points) <= 0
    prediction = analysis._predict(analysis.surrogate_model, points)[0] <= 0
    assert np.mean(weights * (truth != prediction)) < 0.1 * reference
    if problem == "two_tail":
        for region in (points[:, 0] > 4, points[:, 0] < -4):
            assert np.mean(weights * region * (~prediction)) < 0.05 * reference
    elif problem == "four_branch":
        along, across = (points[:, 0] + points[:, 1]) / np.sqrt(2), (
            points[:, 0] - points[:, 1]
        ) / np.sqrt(2)
        for region in (
            along > 3 + 0.2 * across**2,
            along < -3 - 0.2 * across**2,
            across > 3,
            across < -3,
        ):
            assert np.count_nonzero(region) > 50
            assert np.mean(~prediction[region]) < 0.15


@pytest.mark.parametrize("method", ["nataf", "rosenblatt"])
def test_adaptive_subset_respects_dependent_physical_marginals(method):
    model = ra.StochasticModel(
        ra.JointDistribution(
            [ra.Lognormal("x", 2, 0.4), ra.Lognormal("y", 3, 0.6)],
            ra.GaussianCopula([[1, 0.4], [0.4, 1]]),
        )
    )
    options = ra.SimulationOptions(transform=method)
    analysis = ActiveLearning(
        model=model,
        options=options,
        limit_state=ra.LimitState(lambda x, y: 3.3 - np.log(x * y)),
        surrogate="pce",
        surrogate_kwargs={"degree": 1},
        estimator=SubsetSimulationEstimator(),
        target_cov=0.3,
        rng=21,
    )
    result = analysis.run()
    assert result.converged
    points = np.random.default_rng(3).normal(size=(1000, 2))
    mean, std = analysis.surrogate_model.predict(points)
    np.testing.assert_allclose(mean, analysis._evaluate(points), atol=1e-10)
    assert std.max() < 1e-10
