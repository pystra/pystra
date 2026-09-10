"""Analytic and independently integrated active-learning benchmarks.

Normal sum and lognormal beam: benchmarks also used by UQLab, with exact
references derived here from their distributions. Four-branch k=6 problem:
Schueremans & Van Gemert (2005), doi:10.1016/j.strusafe.2004.11.001.
The rotated independent coordinates give a one-dimensional quadrature.
"""

from dataclasses import FrozenInstanceError
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
import pystra as ra
from pystra.active_learning import (
    ActiveLearning,
    KrigingSurrogate,
    PCESurrogate,
    Surrogate,
    learning_eff,
    learning_u,
)


def benchmark(name):
    model = ra.StochasticModel()
    if name == "normal_sum":
        for i in range(5):
            model.add_variable(ra.Normal(f"x{i}", 1, 1))
        function = lambda x0, x1, x2, x3, x4: x0 + x1 + x2 + x3 + x4
        reference = norm.cdf(-np.sqrt(5))
    elif name == "four_branch":
        model.add_variable(ra.Normal("x0", 0, 1))
        model.add_variable(ra.Normal("x1", 0, 1))

        def function(x0, x1):
            along = (x0 + x1) / np.sqrt(2)
            across = (x0 - x1) / np.sqrt(2)
            return np.minimum(
                3 + 0.2 * across**2 - np.abs(along), np.sqrt(2) * (3 - np.abs(across))
            )

        # Failure: |across|>=3 OR |along|>=3+0.2*across**2.
        reference = (
            2 * norm.sf(3)
            + quad(
                lambda v: 2 * norm.sf(3 + 0.2 * v * v) * norm.pdf(v),
                -3,
                3,
                epsabs=1e-13,
            )[0]
        )
    elif name == "beam":
        means = np.array([0.15, 0.3, 5, 30000, 0.01])
        stds = np.array([0.0075, 0.015, 0.05, 4500, 0.002])
        for variable, mean, std in zip(
            ("b", "h", "length", "elasticity", "load"), means, stds
        ):
            model.add_variable(ra.Lognormal(variable, mean, std))
        function = (
            lambda b, h, length, elasticity, load: 0.015
            - 5 * load * length**4 / (32 * elasticity * b * h**3)
        )
        # log(deflection) is exactly normal; no fitted reference probability.
        powers = np.array([-1, -3, 4, -1, 1])
        variances = np.log1p((stds / means) ** 2)
        log_mean = np.log(5 / 32) + powers @ (np.log(means) - variances / 2)
        log_std = np.sqrt(powers**2 @ variances)
        reference = norm.sf((np.log(0.015) - log_mean) / log_std)
    else:
        raise ValueError(name)
    return model, ra.LimitState(function), reference


@pytest.mark.parametrize("mean", [-12, -3, -0.3, 0, 0.3, 3, 12])
@pytest.mark.parametrize("std", [0.01, 1, 7])
def test_eff_against_integrated_triangular_payoff(mean, std):
    # Integrate in standardized coordinates, independently of closed form.
    ratio = mean / std
    expected = quad(
        lambda z: std * (2 - abs(z + ratio)) * norm.pdf(z),
        -2 - ratio,
        2 - ratio,
        points=[-ratio],
        epsabs=1e-13,
    )[0]
    values, _, _ = learning_eff([mean], [std])
    assert values[0] == pytest.approx(expected, rel=1e-9, abs=1e-13)
    assert values[0] == pytest.approx(learning_eff([-mean], [std])[0][0])


def test_learning_limits_validation_and_scaling():
    assert learning_u([1, 0], [0, 0])[0].tolist() == [np.inf, 0]
    assert learning_eff([1, 0], [0, 0])[0].tolist() == [0, 0]
    assert learning_eff([0], [1])[0][0] == pytest.approx(1.219096844430794, abs=1e-12)
    assert learning_eff([6], [2])[0][0] == pytest.approx(
        2 * learning_eff([3], [1])[0][0]
    )
    for function in (learning_u, learning_eff):
        for mean, std in [([], []), ([0], [-1]), ([np.nan], [1]), ([1, 2], [1])]:
            with pytest.raises(ValueError):
                function(mean, std)


def test_pce_polynomial_exactness_local_spread_and_repeatability():
    rng = np.random.default_rng(13)
    points = rng.normal(size=(70, 2))
    values = 1 + points[:, 0] - 0.3 * points[:, 1] ** 2
    surrogate = PCESurrogate(degree=2, method="ols", seed=4)
    surrogate.fit(points, values)
    query = rng.normal(size=(40, 2))
    mean, std = surrogate.predict(query)
    np.testing.assert_allclose(
        mean, 1 + query[:, 0] - 0.3 * query[:, 1] ** 2, atol=1e-12
    )
    assert std.max() < 1e-12
    # Deliberately omitted higher terms create spatially varying fit uncertainty.
    surrogate.fit(points, values + 0.1 * points[:, 0] ** 3)
    first = surrogate.predict(query)
    assert np.ptp(first[1]) > 0.01
    surrogate.fit(points, values + 0.1 * points[:, 0] ** 3)
    np.testing.assert_array_equal(first, surrogate.predict(query))
    with pytest.raises(ValueError, match="more training"):
        surrogate.fit(points[:3], values[:3])
    with pytest.raises(RuntimeError):
        surrogate.predict(query)
    with pytest.raises(ValueError, match="rank deficient"):
        surrogate.fit(np.ones((30, 2)), np.ones(30))


@pytest.mark.parametrize("seed", [7, 23, 101])
@pytest.mark.parametrize(
    "problem,surrogate,learning",
    [
        ("normal_sum", "kriging", "u"),
        ("normal_sum", "kriging", "eff"),
        ("four_branch", "kriging", "u"),
        ("four_branch", "kriging", "eff"),
        ("beam", "kriging", "u"),
        ("normal_sum", "pce", "u"),
        ("beam", "pce", "u"),
        ("four_branch", "pce", "u"),
    ],
)
def test_standard_benchmarks(problem, surrogate, learning, seed):
    if surrogate == "kriging":
        pytest.importorskip("sklearn")
    model, limit_state, reference = benchmark(problem)
    settings = {"n_restarts": 0} if surrogate == "kriging" else {}
    nonsmooth_pce = problem == "four_branch" and surrogate == "pce"
    if nonsmooth_pce:
        settings = {
            "degree": tuple(range(2, 13)),
            "q_norm": (0.75, 1.0),
            "degree_early_stop": False,
        }
    analysis = ActiveLearning(
        stochastic_model=model,
        limit_state=limit_state,
        surrogate=surrogate,
        learning_function=learning,
        surrogate_kwargs=settings,
        n_initial=50 if nonsmooth_pce else None,
        learning_threshold=3 if nonsmooth_pce else None,
        n_candidates=12_000,
        n_estimation=100_000,
        seed=seed,
        max_iterations=180,
    )
    result = analysis.run()
    assert result.converged, result
    assert result.n_evaluations < 250
    # 4 binomial standard errors plus a separate 5% surrogate error allowance.
    sampling_error = 4 * np.sqrt(reference * (1 - reference) / result.n_estimation)
    assert (
        abs(result.failure_probability - reference) < sampling_error + 0.05 * reference
    )
    # Independent point-wise validation detects cancellation in Pf errors.
    points = np.random.default_rng(892).normal(size=(100_000, model.n_marg))
    truth = analysis._evaluate(points) <= 0
    prediction = analysis._predict(analysis.surrogate_model, points)[0] <= 0
    # A global polynomial approximates this nonsmooth minimum less closely
    # than Kriging: permit at most 15% of Pf in misclassified probability
    # mass (10% for the other cases), separately from the same Pf criterion.
    classification_fraction = 0.15 if nonsmooth_pce else 0.1
    assert np.mean(truth != prediction) < classification_fraction * reference
    if problem == "four_branch":
        along = (points[:, 0] + points[:, 1]) / np.sqrt(2)
        across = (points[:, 0] - points[:, 1]) / np.sqrt(2)
        for region in (
            along > 3 + 0.2 * across**2,
            along < -3 - 0.2 * across**2,
            across > 3,
            across < -3,
        ):
            assert np.count_nonzero(region) > 50
            assert np.mean(~prediction[region]) < 0.15


class ExactLinear(Surrogate):
    def __init__(self):
        self.designs = []

    def fit(self, points, values):
        self.designs.append(points.copy())

    def predict(self, points):
        return 2 - points[:, 0], np.zeros(len(points))


def test_result_snapshot_seed_and_independent_estimation():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("x", 0, 1))
    surrogate = ExactLinear()
    analysis = ActiveLearning(
        stochastic_model=model,
        limit_state=ra.LimitState(lambda x: 2 - x),
        surrogate=surrogate,
        seed=12,
    )
    state = np.random.get_state()
    result = analysis.run()
    assert result.converged
    assert result.n_evaluations == 12
    assert result == analysis.run()
    assert result is not analysis.result
    assert np.array_equal(state[1], np.random.get_state()[1])
    with pytest.raises(FrozenInstanceError):
        result.converged = False
    analysis.limitstate = ra.LimitState(lambda x: np.full_like(x, np.nan))
    with pytest.raises(ValueError):
        analysis.run()
    assert analysis.result is None and not analysis.results_valid


def test_budget_and_candidate_exhaustion_do_not_repeat_points():
    class Uncertain(ExactLinear):
        def predict(self, points):
            return np.ones(len(points)), np.ones(len(points))

    model = ra.StochasticModel()
    model.add_variable(ra.Normal("x", 0, 1))
    surrogate = Uncertain()
    analysis = ActiveLearning(
        stochastic_model=model,
        limit_state=ra.LimitState(lambda x: x),
        surrogate=surrogate,
        n_candidates=3,
        max_iterations=5,
        seed=1,
    )
    with pytest.warns(RuntimeWarning, match="candidate_exhaustion"):
        result = analysis.run()
    assert not result.converged and not analysis.results_valid
    assert np.isinf(result.sampling_cov)
    assert result.sampling_interval[1] > 0
    assert len(np.unique(surrogate.designs[-1], axis=0)) == 15
    analysis.max_iterations = 0
    with pytest.warns(RuntimeWarning, match="max_iterations"):
        assert not analysis.run().converged


def test_missing_optional_dependency_does_not_break_core():
    code = """
import sys
sys.modules['sklearn'] = None
import pystra
from pystra.active_learning import KrigingSurrogate, PCESurrogate
PCESurrogate(method="ols")
try:
    KrigingSurrogate()
except ImportError as exc:
    assert 'pystra[al]' in str(exc)
else:
    raise AssertionError('Kriging should require the extra')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_eff_extreme_standardized_tail_is_finite():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values, _, _ = learning_eff([1.0, -1.0], [1e-300, 1e-300])
    np.testing.assert_array_equal(values, [0, 0])


@pytest.mark.parametrize("method", ["nataf", "rosenblatt"])
def test_pce_correlated_lognormals_use_independent_normal_coordinates(method):
    correlation = np.array([[1, 0.4], [0.4, 1]])
    marginals = [ra.Lognormal("x", 2, 0.4), ra.Lognormal("y", 3, 0.6)]
    model = ra.StochasticModel(
        ra.JointDistribution(marginals, ra.GaussianCopula(correlation))
    )
    options = ra.AnalysisOptions()
    options.set_transform(method)
    # The log of a correlated lognormal product is exactly linear in u.
    analysis = ActiveLearning(
        stochastic_model=model,
        analysis_options=options,
        limit_state=ra.LimitState(lambda x, y: 2.6 - np.log(x * y)),
        surrogate="pce",
        surrogate_kwargs={"degree": 1},
        seed=21,
    )
    result = analysis.run()
    assert result.converged
    points = np.random.default_rng(3).normal(size=(1000, 2))
    prediction, spread = analysis.surrogate_model.predict(points)
    np.testing.assert_allclose(prediction, analysis._evaluate(points), atol=1e-10)
    assert spread.max() < 1e-10


def test_spherical_student_space_is_rejected():
    model = ra.StochasticModel(
        ra.JointDistribution(
            [ra.Normal("x", 0, 1), ra.Normal("y", 0, 1)],
            ra.StudentTCopula(np.eye(2), 4),
        )
    )
    options = ra.AnalysisOptions()
    options.set_transform("nataf")
    analysis = ActiveLearning(
        stochastic_model=model,
        analysis_options=options,
        limit_state=ra.LimitState(lambda x, y: 3 - x - y),
        surrogate="pce",
    )
    with pytest.raises(ValueError, match="independent normal space"):
        analysis.run()
    assert analysis.result is None


def test_final_sampling_precision_is_separate_from_learning():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("x", 0, 1))
    analysis = ActiveLearning(
        stochastic_model=model,
        limit_state=ra.LimitState(lambda x: 2 - x),
        surrogate=ExactLinear(),
        n_estimation=10,
        seed=1,
    )
    with pytest.warns(RuntimeWarning, match="sampling_precision"):
        result = analysis.run()
    assert not result.converged
    assert result.sampling_interval[1] > result.failure_probability
