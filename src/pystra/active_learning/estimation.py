"""Final reliability estimation conditional on a frozen surrogate."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.stats import binomtest

from ._validation import _positive_integer, _predictions
from .results import ReliabilityEstimate


class ReliabilityEstimator(ABC):
    """Stateless final estimator using independent standard normal coordinates.

    The supplied predictor accepts row-wise (n_samples, n_variables) points
    and returns mean/std arrays. Failure is mean <= 0. It does not evaluate
    the true model. Use only the run-owned ``rng`` for randomness, and keep
    samples separate from training/enrichment. Estimators own their sample
    weights and dependence-aware uncertainty calculations: the runner never
    replaces those diagnostics with a binomial formula.

    This interface controls the final estimate. The current runner's fixed
    Monte Carlo enrichment pool is a separate restriction; substituting a
    final estimator alone does not implement active subset simulation or IS.
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
