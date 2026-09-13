"""Active importance sampling with explicit Gaussian mixture proposals.

Design-point AK-IS: Echard et al. (2013), doi:10.1016/j.ress.2012.10.008.
This variant supports multiple supplied centers and a defensive normal component.
It does not automatically discover failure modes or apply a true-model correction.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

from ._validation import _positive_integer, _points, _predictions
from .estimation import EnrichmentEstimator, EnrichmentResult
from .results import ReliabilityEstimate

__all__ = ["ImportanceSamplingDiagnostics", "ImportanceSamplingEstimator"]


@dataclass(frozen=True)
class ImportanceSamplingDiagnostics:
    """IID likelihood-weight diagnostics conditional on a frozen surrogate.

    ``effective_failures`` uses failure contributions w*I, while
    ``effective_samples`` uses all weights. Neither detects unsampled modes.
    ``raw_probability`` retains an out-of-range estimate if a finite sample
    cannot give a valid probability; such results are explicitly incomplete.
    """

    raw_probability: float
    standard_error: float
    effective_samples: float
    effective_failures: float
    failure_count: int
    mean_weight: float
    max_weight: float


class ImportanceSamplingEstimator(EnrichmentEstimator):
    """Independent Gaussian-mixture importance sampling for active reliability.

    Parameters
    ----------
    centers : array_like
        Explicit row-wise proposal centers in independent normal coordinates,
        shape (n_centers, n_variables). A FORM standard_point is one useful
        center. Supply all relevant centers for separated failure regions.
    scale : float
        Common spherical proposal standard deviation, default 1.
    defensive_fraction : float
        Mixture weight of the target N(0,I), default 0.1, in [0,1).
        A positive fraction bounds likelihood ratios by its reciprocal but
        cannot guarantee discovery of an unrepresented rare failure mode.
    n_samples : int
        IID mixture draws per exploration/final estimate, default 10000.
    min_effective_failures : float
        Minimum effective failure contributions for a completed estimate,
        default 20. The final stopping policy additionally checks sampling CoV.

    Notes
    -----
    Uses the ordinary estimator mean(I*f/q), not a self-normalized ratio.
    Sampling SE is the sample standard deviation of I*f/q divided by sqrt(N).
    No binomial interval is reported. Bands use the same likelihood weights
    for mean +/- 2 std events and are clipped to probability limits only as
    diagnostics. An out-of-range point estimate is retained in diagnostics,
    exposed at the nearest endpoint and marked invalid with infinite CoV.
    Zero/all observed failures and inadequate effective failures are incomplete.
    Proposal centers remain fixed during a run; learning adapts the surrogate.
    Exploration reuses common random numbers; final estimation uses fresh draws.
    """

    def __init__(
        self,
        *,
        centers: np.ndarray,
        scale: float = 1.0,
        defensive_fraction: float = 0.1,
        n_samples: int = 10_000,
        min_effective_failures: float = 20,
    ) -> None:
        centers = _points(centers)
        self.centers = tuple(tuple(float(value) for value in row) for row in centers)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not np.isfinite(defensive_fraction) or not 0 <= defensive_fraction < 1:
            raise ValueError("defensive_fraction must be in [0, 1)")
        if not np.isfinite(min_effective_failures) or min_effective_failures < 2:
            raise ValueError("min_effective_failures must be finite and at least 2")
        self.scale = float(scale)
        self.defensive_fraction = float(defensive_fraction)
        self.n_samples = _positive_integer(n_samples, "n_samples", 2)
        self.min_effective_failures = float(min_effective_failures)

    def _sample(self, dimension, rng):
        dimension = _positive_integer(dimension, "dimension")
        centers = np.asarray(self.centers)
        if centers.shape[1] != dimension:
            raise ValueError("Proposal dimension differs from the stochastic model")
        fractions = np.r_[
            self.defensive_fraction,
            np.full(len(centers), (1 - self.defensive_fraction) / len(centers)),
        ]
        choices = rng.choice(len(centers) + 1, size=self.n_samples, p=fractions)
        points = rng.standard_normal((self.n_samples, dimension))
        shifted = choices > 0
        points[shifted] = points[shifted] * self.scale + centers[choices[shifted] - 1]
        log_target = -0.5 * np.sum(points**2, axis=1) - dimension / 2 * np.log(
            2 * np.pi
        )
        log_shifted = (
            -0.5 * cdist(points / self.scale, centers / self.scale, "sqeuclidean")
            - dimension * np.log(self.scale)
            - dimension / 2 * np.log(2 * np.pi)
            + np.log((1 - self.defensive_fraction) / len(centers))
        )
        terms = log_shifted
        if self.defensive_fraction:
            terms = np.column_stack(
                (log_target + np.log(self.defensive_fraction), terms)
            )
        with np.errstate(over="ignore", invalid="ignore"):
            weights = np.exp(log_target - logsumexp(terms, axis=1))
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError(
                "Importance weights are not finite; revise the proposal"
            )
        return points, weights

    def _calculate(self, predict, dimension, rng):
        points, weights = self._sample(dimension, rng)
        mean, std = _predictions(*predict(points))
        if len(mean) != len(points):
            raise ValueError("Predictor returned the wrong number of predictions")
        failure = mean <= 0
        contributions = weights * failure
        probability = float(contributions.mean())
        error = float(contributions.std(ddof=1) / np.sqrt(self.n_samples))

        def effective(values):
            maximum = values.max()
            if maximum == 0:
                return 0.0
            scaled = values / maximum
            return float(scaled.sum() ** 2 / (scaled @ scaled))

        effective_failures = effective(contributions)
        status = "completed"
        if not 0 <= probability <= 1:
            status = "invalid_probability"
        elif not failure.any():
            status = "no_failures"
        elif failure.all():
            status = "all_failures"
        elif effective_failures < self.min_effective_failures:
            status = "insufficient_effective_failures"
        if status == "completed" and not 0 < probability < 1:
            status = "probability_endpoint"
        complete = status == "completed"
        diagnostic = ImportanceSamplingDiagnostics(
            probability,
            error,
            effective(weights),
            effective_failures,
            int(failure.sum()),
            float(weights.mean()),
            float(weights.max()),
        )
        estimate = ReliabilityEstimate(
            failure_probability=float(np.clip(probability, 0, 1)),
            sampling_cov=error / probability if complete else np.inf,
            sampling_interval=None,
            confidence_level=None,
            n_samples=self.n_samples,
            method="importance_sampling",
            sampling_dependence="independent",
            converged=complete,
            status=status,
            diagnostics=(diagnostic,),
        )
        with np.errstate(over="ignore"):
            band = tuple(
                np.clip(
                    [
                        np.mean(weights * (mean + 2 * std <= 0)),
                        np.mean(weights * (mean - 2 * std <= 0)),
                    ],
                    0,
                    1,
                )
            )
        return points, mean, std, estimate, band

    def estimate(
        self,
        predict: Callable[[np.ndarray], tuple],
        *,
        dimension: int,
        rng: np.random.Generator,
    ) -> ReliabilityEstimate:
        """Classify fresh IID proposal draws and retain weighted uncertainty."""
        return self._calculate(predict, dimension, rng)[3]

    def explore(
        self,
        predict: Callable[[np.ndarray], tuple],
        *,
        dimension: int,
        rng: np.random.Generator,
        n_candidates: int,
    ) -> EnrichmentResult:
        """Supply proposal candidates and correctly weighted probability bands."""
        n_candidates = _positive_integer(n_candidates, "n_candidates")
        points, mean, std, estimate, band = self._calculate(predict, dimension, rng)
        selection = (
            rng.choice(len(points), n_candidates, replace=False)
            if n_candidates < len(points)
            else np.arange(len(points))
        )
        return EnrichmentResult(
            points[selection], mean[selection], std[selection], estimate, band
        )
