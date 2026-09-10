"""Final reliability estimation conditional on a frozen surrogate."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import binomtest

from ._validation import _positive_integer, _predictions, _points
from .results import ReliabilityEstimate

__all__ = [
    "ReliabilityEstimator",
    "EnrichmentResult",
    "EnrichmentEstimator",
    "MonteCarloEstimator",
]


class ReliabilityEstimator(ABC):
    """Stateless final estimator using independent standard normal coordinates.

    The supplied predictor accepts row-wise (n_samples, n_variables) points
    and returns mean/std arrays. Failure is mean <= 0. It does not evaluate
    the true model. Use only the run-owned ``rng`` for randomness, and keep
    samples separate from training/enrichment. Estimators own their sample
    weights and dependence-aware uncertainty calculations: the runner never
    replaces those diagnostics with a binomial formula.

    This base interface controls the final estimate, retaining fixed normal
    MC enrichment. Subclass EnrichmentEstimator to also generate enrichment
    populations and probability diagnostics under the estimator's measure.
    """

    @abstractmethod
    def estimate(
        self,
        predict: Callable[[np.ndarray], tuple],
        *,
        dimension: int,
        rng: np.random.Generator,
    ) -> ReliabilityEstimate:
        """Estimate conditional Pf from a frozen surrogate, returning a snapshot."""


@dataclass(frozen=True)
class EnrichmentResult:
    """Estimator-generated points and probability diagnostics for one fit.

    ``points`` are row-wise independent-normal coordinates, with ``mean`` and
    ``std`` predictions at those points. Their sampling distribution may be
    conditional or weighted: they are used for selection only. ``estimate``
    and the ascending ``probability_band`` must be computed by the estimator
    under its sampling measure, never by averaging these pooled predictions.
    The band describes mean +/- 2 std, not true-model confidence limits.
    Arrays are copied into read-only buffers; they do not alias sampler state.
    """

    points: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    estimate: ReliabilityEstimate
    probability_band: tuple

    def __post_init__(self):
        points = _points(self.points)
        mean, std = _predictions(self.mean, self.std)
        if len(points) != len(mean):
            raise ValueError("Enrichment predictions must match the point count")
        if not isinstance(self.estimate, ReliabilityEstimate):
            raise TypeError("estimate must be a ReliabilityEstimate")
        band = tuple(float(value) for value in self.probability_band)
        if (
            len(band) != 2
            or not 0 <= band[0] <= self.estimate.failure_probability <= band[1] <= 1
        ):
            raise ValueError("probability_band must enclose the estimate in [0, 1]")
        object.__setattr__(self, "probability_band", band)
        for name, values in (("points", points), ("mean", mean), ("std", std)):
            frozen = np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)
            object.__setattr__(self, name, frozen)


class EnrichmentEstimator(ReliabilityEstimator):
    """Estimator that also constructs a new learning population after each fit.

    The runner restarts an exploration RNG from the same seed for each fit
    (common random numbers), and uses a separate stream for final estimation.
    Implementations must not retain mutable run state. Sampling weights and
    dependence belong in returned diagnostics, not in the learning function.
    """

    @abstractmethod
    def explore(
        self,
        predict: Callable[[np.ndarray], tuple],
        *,
        dimension: int,
        rng: np.random.Generator,
        n_candidates: int,
    ) -> EnrichmentResult:
        """Return at most n_candidates selectable points and fit diagnostics."""


class MonteCarloEstimator(ReliabilityEstimator):
    """Independent normal Monte Carlo with an exact 95% binomial interval.

    ``n_samples`` defaults to 100000 and must be at least 2. No/all failures
    yield infinite CoV, so endpoint samples cannot claim adequate precision.
    All diagnostics are conditional on the supplied surrogate.
    """

    def __init__(self, *, n_samples: int = 100_000):
        self.n_samples = _positive_integer(n_samples, "n_samples", 2)

    def estimate(
        self,
        predict: Callable[[np.ndarray], tuple],
        *,
        dimension: int,
        rng: np.random.Generator,
    ) -> ReliabilityEstimate:
        """Classify fresh row-wise normal points using the supplied predictor."""
        dimension = _positive_integer(dimension, "dimension")
        points = rng.standard_normal((self.n_samples, dimension))
        mean, _ = _predictions(*predict(points))
        if len(mean) != self.n_samples:
            raise ValueError("Predictor returned the wrong number of predictions")
        failures = int(np.sum(mean <= 0))
        probability = failures / self.n_samples
        cov = (
            np.sqrt((1 - probability) / (self.n_samples * probability))
            if 0 < failures < self.n_samples
            else np.inf
        )
        interval = binomtest(failures, self.n_samples).proportion_ci(
            confidence_level=0.95
        )
        return ReliabilityEstimate(
            failure_probability=probability,
            sampling_cov=float(cov),
            sampling_interval=(float(interval.low), float(interval.high)),
            confidence_level=0.95,
            n_samples=self.n_samples,
            method="monte_carlo",
            sampling_dependence="independent",
        )
