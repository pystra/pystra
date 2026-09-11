"""Component contracts, published stopping rules and convergence failure modes."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.stats import norm

import pystra as ra
from pystra.active_learning import (
    ActiveLearning,
    AllCriteria,
    BetaBounds,
    BetaStability,
    ExpectedFeasibility,
    LearningDecision,
    LearningFunction,
    LearningStep,
    LearningThreshold,
    MonteCarloEstimator,
    ReliabilityEstimate,
    ReliabilityEstimator,
    UFunction,
)
from .test_active_learning import ExactLinear, benchmark


def linear_analysis(**settings):
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("x", 0, 1))
    return ActiveLearning(
        model=model,
        limit_state=ra.LimitState(lambda x: 2 - x),
        surrogate=ExactLinear(),
        rng=12,
        **settings,
    )


def step(beta=3.0, band=(2.99, 3.01), satisfied=True):
    return LearningStep(
        failure_probability=float(norm.sf(beta)),
        learning_score=2.5,
        n_limit_state_evaluations=20,
        probability_band=(float(norm.sf(band[1])), float(norm.sf(band[0]))),
        beta_band=band,
        learning_satisfied=satisfied,
    )


def test_named_shortcuts_and_explicit_components_have_identical_results():
    shortcut = linear_analysis(
        n_estimation=20_000, target_cov=0.2, learning_threshold=3
    )
    explicit = linear_analysis(
        estimator=MonteCarloEstimator(n_samples=20_000),
        stopping_criterion=LearningThreshold(target_cov=0.2),
        learning_function=UFunction(threshold=3),
    )
    assert shortcut.run() == explicit.run()
    assert explicit.result.estimate.method == "monte_carlo"
    assert explicit.result.estimate.sampling_dependence == "independent"
    assert explicit.result.estimate.confidence_level == 0.95
    first = explicit.result.history[0]
    assert first.probability_band == (first.failure_probability,) * 2
    assert first.beta_band == (first.beta,) * 2


def test_review_beta_bounds_requires_three_successive_fits_and_resets():
    criterion = BetaBounds()
    good = step()
    wide = step(band=(2.9, 3.1))
    assert not criterion.should_stop((good, good))
    assert criterion.should_stop((good,) * 3)
    assert not criterion.should_stop((good, good, wide))
    assert not criterion.should_stop((good, wide, good, good))
    assert criterion.should_stop((wide, good, good, good))
    # Learning-score convergence is a different decision, not a hidden gate.
    assert criterion.should_stop((replace(good, learning_satisfied=False),) * 3)


def test_review_beta_stability_counts_changes_not_fits():
    criterion = BetaStability()
    good = step()
    jump = step(beta=3.1)
    assert not criterion.should_stop((good,) * 3)
    assert criterion.should_stop((good,) * 4)
    assert not criterion.should_stop((good, good, good, jump))
    assert not criterion.should_stop((jump, good, good, good))
    assert criterion.should_stop((jump, good, good, good, good))
    # Stability alone deliberately does not certify the prediction band.
    wide = step(band=(1, 5))
    assert criterion.should_stop((wide,) * 4)
    assert not AllCriteria(criteria=(criterion, BetaBounds())).should_stop((wide,) * 4)


@pytest.mark.parametrize("beta", [np.inf, -np.inf, 0])
def test_beta_criteria_do_not_divide_by_zero_or_accept_unresolved_tails(beta):
    history = (step(beta=beta, band=(beta, beta)),) * 4
    with np.errstate(all="raise"):
        assert not BetaBounds().should_stop(history)
        assert not BetaStability().should_stop(history)


def test_negative_beta_uses_magnitude_in_relative_tolerance():
    wide = step(beta=-3, band=(-4, -2))
    assert not BetaBounds().should_stop((wide,) * 3)
    assert not BetaStability(consecutive=1).should_stop((step(beta=-2), wide))
    narrow = step(beta=-3, band=(-3.01, -2.99))
    assert BetaBounds().should_stop((narrow,) * 3)


def test_combined_stopping_refits_then_starts_fresh_on_rerun():
    # The review's combined policy uses two consecutive band/stability tests.
    policy = AllCriteria(
        criteria=(BetaBounds(consecutive=2), BetaStability(consecutive=2))
    )
    analysis = linear_analysis(stopping_criterion=policy)
    result = analysis.run()
    assert result.converged
    assert len(result.history) == 3
    assert result.n_limit_state_evaluations == 14
    assert result == analysis.run()
    with pytest.raises(FrozenInstanceError):
        result.history[0].probability_band = (0, 1)
    with pytest.raises(FrozenInstanceError):
        result.estimate.sampling_cov = 0
    analysis.max_iterations = 1
    with pytest.warns(RuntimeWarning, match="max_iterations"):
        assert not analysis.run().converged
    assert result.converged


def test_beta_stability_cannot_turn_all_safe_predictions_into_convergence():
    analysis = linear_analysis(stopping_criterion=BetaStability(), max_iterations=4)
    analysis.surrogate.predict = lambda points: (
        np.ones(len(points)),
        np.zeros(len(points)),
    )
    with pytest.warns(RuntimeWarning, match="max_iterations"):
        result = analysis.run()
    assert not result.converged
    assert np.isinf(result.sampling_cov)
    assert result.sampling_interval[1] > 0


class FirstPoint(LearningFunction):
    def select(self, mean, std):
        return LearningDecision(0, 42.0, True)


class DependentEstimator(ReliabilityEstimator):
    def __init__(self, cov=0.3):
        self.cov = cov

    def estimate(self, predict, *, dimension, rng):
        # A contract double, not an implementation of correlated simulation.
        # Deliberately different precision from the IID binomial formula.
        points = rng.standard_normal((64, dimension))
        mean, std = predict(points)
        np.testing.assert_array_equal(mean, 2 - points[:, 0])
        np.testing.assert_array_equal(std, np.zeros(64))
        return ReliabilityEstimate(
            failure_probability=0.02,
            sampling_cov=self.cov,
            sampling_interval=None,
            confidence_level=None,
            n_samples=64,
            method="contract_test",
            sampling_dependence="dependent",
        )


def test_custom_estimator_owns_uncertainty_and_selection_owns_enrichment():
    analysis = linear_analysis(
        learning_function=FirstPoint(),
        estimator=DependentEstimator(),
        stopping_criterion=AllCriteria(
            criteria=(LearningThreshold(target_cov=0.5), BetaBounds(target_cov=0.1))
        ),
    )
    with pytest.warns(RuntimeWarning, match="sampling_precision"):
        result = analysis.run()
    assert result.n_estimation == 64
    assert result.sampling_cov == 0.3
    assert result.sampling_interval is None
    assert result.estimate.sampling_dependence == "dependent"
    assert all(item.learning_score == 42 for item in result.history)
    assert result.n_limit_state_evaluations == 14
    analysis.estimator.cov = 0.05
    assert analysis.run().converged

    def broken(predict, **kwargs):
        raise RuntimeError("estimator failed")

    analysis.estimator.estimate = broken
    with pytest.raises(RuntimeError, match="estimator failed"):
        analysis.run()
    assert analysis.result is None and not analysis._results_valid
    assert analysis.surrogate_model is None


def test_final_estimation_does_not_reuse_training_or_candidate_points():
    analysis = linear_analysis()
    seen = []
    original = analysis.surrogate.predict

    def predict(points):
        seen.append(points.copy())
        return original(points)

    analysis.surrogate.predict = predict
    analysis.run()
    predicted = np.vstack(seen)
    assert len(predicted) == 110_000
    assert len(np.unique(predicted, axis=0)) == len(predicted)
    combined = np.vstack((predicted, analysis.surrogate.designs[0]))
    assert len(np.unique(combined, axis=0)) == len(combined)


@pytest.mark.parametrize("seed", [7, 23, 101])
def test_monte_carlo_estimator_against_exact_linear_probability(seed):
    estimate = MonteCarloEstimator().estimate(
        lambda points: (2 - points[:, 0], np.zeros(len(points))),
        dimension=1,
        rng=np.random.default_rng(seed),
    )
    reference = norm.sf(2)
    assert abs(estimate.failure_probability - reference) < 4 * np.sqrt(
        reference * (1 - reference) / estimate.n_samples
    )
    assert (
        estimate.sampling_interval[0]
        < estimate.failure_probability
        < estimate.sampling_interval[1]
    )


@pytest.mark.parametrize("value", [0, 1, -1])
def test_monte_carlo_includes_failure_boundary_and_rejects_endpoint_precision(value):
    estimate = MonteCarloEstimator(n_samples=10).estimate(
        lambda points: (np.full(len(points), value), np.zeros(len(points))),
        dimension=1,
        rng=np.random.default_rng(4),
    )
    assert estimate.failure_probability == float(value <= 0)
    assert np.isinf(estimate.sampling_cov)
    assert not LearningThreshold().accepts_estimate(estimate)


@pytest.mark.parametrize(
    "settings",
    [
        {"learning_function": object()},
        {"learning_function": UFunction(), "learning_threshold": 2},
        {"estimator": object()},
        {"estimator": MonteCarloEstimator(), "n_estimation": 100},
        {"stopping_criterion": object()},
        {"stopping_criterion": LearningThreshold(), "target_cov": 0.1},
    ],
)
def test_component_configuration_rejects_ambiguous_or_invalid_settings(settings):
    with pytest.raises(ValueError):
        linear_analysis(**settings)


@pytest.mark.parametrize(
    "factory,settings",
    [
        (MonteCarloEstimator, {"n_samples": 1}),
        (UFunction, {"threshold": np.nan}),
        (ExpectedFeasibility, {"threshold": 0}),
        (LearningThreshold, {"target_cov": np.inf}),
        (BetaBounds, {"tolerance": 0}),
        (BetaStability, {"consecutive": True}),
        (AllCriteria, {"criteria": ()}),
        (AllCriteria, {"criteria": (object(),)}),
    ],
)
def test_invalid_component_parameters(factory, settings):
    with pytest.raises(ValueError):
        factory(**settings)


def test_bad_component_outputs_invalidate_run():
    analysis = linear_analysis(learning_function=FirstPoint())
    analysis.learning_function.select = lambda mean, std: LearningDecision(
        len(mean), 0, False
    )
    with pytest.raises(ValueError, match="outside"):
        analysis.run()
    analysis.learning_function.select = lambda mean, std: (0, 0, False)
    with pytest.raises(TypeError, match="LearningDecision"):
        analysis.run()
    analysis.learning_function = UFunction()
    analysis.estimator.estimate = lambda predict, **kwargs: {
        "failure_probability": 0.02
    }
    with pytest.raises(TypeError, match="ReliabilityEstimate"):
        analysis.run()
    assert analysis.result is None


@pytest.mark.parametrize("seed", [7, 23, 101])
@pytest.mark.parametrize("problem", ["normal_sum", "four_branch"])
def test_combined_beta_policies_on_standard_benchmarks(problem, seed):
    pytest.importorskip("sklearn")
    model, limit_state, reference = benchmark(problem)
    analysis = ActiveLearning(
        model=model,
        limit_state=limit_state,
        surrogate_kwargs={"n_restarts": 0},
        stopping_criterion=AllCriteria(
            criteria=(
                LearningThreshold(),
                BetaBounds(consecutive=2),
                BetaStability(consecutive=2),
            )
        ),
        n_candidates=12_000,
        max_iterations=180,
        rng=seed,
    )
    result = analysis.run()
    assert result.converged
    assert result.n_limit_state_evaluations < 250
    sampling_error = 4 * np.sqrt(reference * (1 - reference) / result.n_estimation)
    assert (
        abs(result.failure_probability - reference) < sampling_error + 0.05 * reference
    )
    points = np.random.default_rng(892).normal(size=(100_000, model.n_marg))
    truth = analysis._evaluate(points) <= 0
    predicted = analysis._predict(analysis.surrogate_model, points)[0] <= 0
    assert np.mean(truth != predicted) < 0.1 * reference
    if problem == "four_branch":
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
            assert np.mean(~predicted[region]) < 0.15
