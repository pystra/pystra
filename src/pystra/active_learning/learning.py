"""Point selection in independent standard normal coordinates."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ._validation import _predictions, _positive_integer

__all__ = [
    "LearningDecision",
    "LearningFunction",
    "UFunction",
    "ExpectedFeasibility",
    "learning_u",
    "learning_eff",
    "EnsembleLearningFunction",
    "FBRLearning",
]


@dataclass(frozen=True)
class LearningDecision:
    """Selection from the supplied pool, with a diagnostic threshold flag.

    ``index`` is relative to the input predictions, not the full candidate
    pool. ``score`` may be infinite (for example U at zero spread), but not
    NaN. The stopping criterion decides whether to use ``threshold_satisfied``.
    """

    index: int
    score: float
    threshold_satisfied: bool

    def __post_init__(self):
        _positive_integer(self.index, "index", 0)
        if np.isnan(self.score):
            raise ValueError("Learning score must not be NaN")
        if not isinstance(self.threshold_satisfied, (bool, np.bool_)):
            raise ValueError("threshold_satisfied must be a bool")


class LearningFunction(ABC):
    """Stateless selection policy operating on scalar mean/spread predictions."""

    @abstractmethod
    def select(self, mean: np.ndarray, std: np.ndarray) -> LearningDecision:
        """Select one point from finite arrays of shape (n_candidates,)."""


class UFunction(LearningFunction):
    """Minimum standardized distance to failure; default threshold is 2.

    See Echard, Gayton and Lemaire (2011). Zero spread has the semantics
    documented in :func:`learning_u`.
    """

    def __init__(self, *, threshold: float = 2.0):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        self.threshold = float(threshold)

    def select(self, mean: np.ndarray, std: np.ndarray) -> LearningDecision:
        """Select the minimum U, reporting whether it reaches the threshold."""
        scores, best, satisfied = learning_u(mean, std, threshold=self.threshold)
        return LearningDecision(best, float(scores[best]), satisfied)


class ExpectedFeasibility(LearningFunction):
    """Maximum EFF with a default threshold of 1e-3 in limit-state units.

    This criterion uses a Gaussian predictive distribution. Applied to
    bootstrap spread, that assumption is a heuristic, not a posterior model.
    See Bichon et al. (2008) and :func:`learning_eff`.
    """

    def __init__(self, *, threshold: float = 1e-3):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        self.threshold = float(threshold)

    def select(self, mean: np.ndarray, std: np.ndarray) -> LearningDecision:
        """Select the maximum EFF, reporting whether it is below tolerance."""
        scores, best, satisfied = learning_eff(mean, std, threshold=self.threshold)
        return LearningDecision(best, float(scores[best]), satisfied)


def learning_u(mean: np.ndarray, std: np.ndarray, *, threshold: float = 2.0) -> tuple:
    """Return the U score, best index (minimum), and threshold satisfaction.

    Zero spread gives infinity away from zero and zero at the boundary.
    Arrays have shape (n_candidates,). ``threshold`` must be positive.
    """
    mean, std = _predictions(mean, std)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    values = np.full_like(mean, np.inf)
    np.divide(np.abs(mean), std, out=values, where=std > 0)
    values[(std == 0) & (mean == 0)] = 0
    best = int(np.argmin(values))
    return values, best, bool(values[best] >= threshold)


def learning_eff(
    mean: np.ndarray, std: np.ndarray, *, threshold: float = 1e-3
) -> tuple:
    r"""Return expected feasibility, best index (maximum), and stopping flag.

    Here G is Gaussian with the supplied mean/std, each (n_candidates,).
    EFF is symmetric in mean and has the units of the limit state.
    ``threshold`` is an absolute, positive tolerance in those units.
    Zero spread gives zero EFF, including at the boundary.
    """
    mean, std = _predictions(mean, std)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    ratio = np.zeros_like(mean)
    with np.errstate(over="ignore"):
        np.divide(np.abs(mean), std, out=ratio, where=std > 0)
    ratio = np.minimum(ratio, 40.0)  # Gaussian payoff underflows beyond this tail
    # Integrate the two linear halves of the triangular feasibility function.
    lower, middle, upper = -2 - ratio, -ratio, 2 - ratio
    values = std * (
        (2 + ratio) * (norm.cdf(middle) - norm.cdf(lower))
        + (2 - ratio) * (norm.cdf(upper) - norm.cdf(middle))
        + norm.pdf(lower)
        + norm.pdf(upper)
        - 2 * norm.pdf(middle)
    )
    values = np.maximum(values, 0.0)  # roundoff only, not an absolute value
    best = int(np.argmax(values))
    return values, best, bool(values[best] < threshold)


class EnsembleLearningFunction(LearningFunction):
    """Selection requiring actual replicate predictions, not Gaussian spread."""

    def select(self, mean: np.ndarray, std: np.ndarray) -> LearningDecision:
        """Reject lossy mean/std input; use select_replicates instead."""
        raise TypeError("This learning function requires replicate predictions")

    @abstractmethod
    def select_replicates(self, predictions: np.ndarray) -> LearningDecision:
        """Select from finite (n_candidates, n_replicates) predictions."""


class FBRLearning(EnsembleLearningFunction):
    """Minimum bootstrap classification agreement, ``abs(B_safe-B_failure)/B``.

    Marelli & Sudret (2018), Eq. (10), doi:10.1016/j.strusafe.2018.06.003.
    Zero is maximal disagreement; one is unanimity. Zero response counts as
    failure. Ties select the first candidate. The threshold flag means all
    candidates are unanimous; it is not a guarantee of surrogate accuracy.
    Use BootstrapBounds for the paper's probability-range stopping rule.
    """

    def select_replicates(self, predictions: np.ndarray) -> LearningDecision:
        """Use actual classification votes, retaining non-Gaussian behaviour."""
        predictions = _replicates(predictions)
        count = predictions.shape[1]
        failures = np.count_nonzero(predictions <= 0, axis=1)
        agreement = np.abs(count - 2 * failures) / count
        index = int(np.argmin(agreement))
        return LearningDecision(
            index, float(agreement[index]), bool(agreement[index] == 1)
        )


def _replicates(predictions):
    predictions = np.asarray(predictions, dtype=float)
    if (
        predictions.ndim != 2
        or not len(predictions)
        or predictions.shape[1] < 2
        or not np.all(np.isfinite(predictions))
    ):
        raise ValueError(
            "Replicate predictions must be finite with shape (n_points, n_replicates>=2)"
        )
    return predictions
