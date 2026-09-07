"""Stateless stopping policies evaluated against immutable fit histories.

Beta criteria follow Eqs. (2)-(3) of Moustapha, Marelli and Sudret (2022),
doi:10.1016/j.strusafe.2021.102174. Prediction bands are surrogate diagnostics,
not rigorous bounds on the true failure probability.
"""

from abc import ABC, abstractmethod

import numpy as np

from ._validation import _positive_integer
from .results import LearningStep, ReliabilityEstimate


class StoppingCriterion(ABC):
    """Separate surrogate convergence from final sampling precision.

    Policies must not keep mutable per-run counters: use the supplied history
    so repeated analyses start fresh. ``target_cov`` defaults to 0.1. The
    default final check rejects zero/all failures and nonfinite CoV, even if
    the surrogate policy is satisfied. Custom policies may override either
    decision and must document the meaning of convergence.
    """

    def __init__(self, *, target_cov: float = 0.1):
        if not np.isfinite(target_cov) or target_cov <= 0:
            raise ValueError("target_cov must be finite and positive")
        self.target_cov = float(target_cov)

    @abstractmethod
    def should_stop(self, history: tuple[LearningStep, ...]) -> bool:
        """Decide surrogate convergence from fits up to and including this fit."""

    def accepts_estimate(self, estimate: ReliabilityEstimate) -> bool:
        """Require an interior probability and adequate finite sampling CoV."""
        return bool(
            0 < estimate.failure_probability < 1
            and np.isfinite(estimate.sampling_cov)
            and estimate.sampling_cov <= self.target_cov
        )


class LearningThreshold(StoppingCriterion):
    """Require the selection threshold and both predicted event classes.

    This is the existing AK-MCS-style default. It uses the selection policy's
    threshold, so choosing a different learning function also changes that
    threshold's meaning. ``target_cov`` defaults to 0.1 for the final sample.
    """

    def should_stop(self, history: tuple[LearningStep, ...]) -> bool:
        """Accept the latest fit only when failure and survival are represented."""
        return bool(
            history
            and history[-1].learning_satisfied
            and 0 < history[-1].failure_probability < 1
        )


class BetaBounds(StoppingCriterion):
    """Require a small relative spread of surrogate-based beta endpoints.

    The criterion is (beta_upper - beta_lower) / abs(beta) <= tolerance for
    ``consecutive`` fits. Defaults (0.01, 3) and the +/-2 std prediction bands
    follow Moustapha et al. (2022), Eq. (2). Nonfinite bands and beta == 0 do
    not pass. Using abs(beta) extends the denominator to probabilities above
    one half; it is unchanged in the paper's positive-beta regime.

    For bootstrap surrogates the band is only a spread sensitivity diagnostic;
    it is not a Gaussian confidence band. ``target_cov`` defaults to 0.1.
    """

    def __init__(
        self, *, tolerance: float = 0.01, consecutive: int = 3, target_cov: float = 0.1
    ):
        super().__init__(target_cov=target_cov)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self.tolerance = float(tolerance)
        self.consecutive = _positive_integer(consecutive, "consecutive")

    def should_stop(self, history: tuple[LearningStep, ...]) -> bool:
        """Require the beta-band test on every fit in the trailing window."""
        if len(history) < self.consecutive:
            return False
        for step in history[-self.consecutive :]:
            beta = step.beta
            lower, upper = step.beta_band
            if not np.all(np.isfinite((beta, lower, upper))) or beta == 0:
                return False
            if (upper - lower) / abs(beta) > self.tolerance:
                return False
        return True


class BetaStability(StoppingCriterion):
    """Require small relative changes in beta across consecutive enrichments.

    The criterion is abs(beta_current - beta_previous) / abs(beta_current)
    <= tolerance. Defaults (0.005, 3 consecutive changes) follow Moustapha
    et al. (2022), Eq. (3), requiring at least four fits. Nonfinite indices
    and current beta == 0 do not pass. Absolute denominators also support
    negative beta. ``target_cov`` defaults to 0.1 for the final sample.

    A stable biased surrogate may pass. Combine this policy with BetaBounds
    or LearningThreshold and independently validate the true failure region.
    """

    def __init__(
        self, *, tolerance: float = 0.005, consecutive: int = 3, target_cov: float = 0.1
    ):
        super().__init__(target_cov=target_cov)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self.tolerance = float(tolerance)
        self.consecutive = _positive_integer(consecutive, "consecutive")

    def should_stop(self, history: tuple[LearningStep, ...]) -> bool:
        """Require the relative-change test on every consecutive fit pair."""
        if len(history) <= self.consecutive:
            return False
        window = history[-self.consecutive - 1 :]
        for previous, current in zip(window, window[1:]):
            beta = current.beta
            if not np.all(np.isfinite((beta, previous.beta))) or beta == 0:
                return False
            if abs(beta - previous.beta) / abs(beta) > self.tolerance:
                return False
        return True


class AllCriteria(StoppingCriterion):
    """Require every supplied policy's surrogate and sampling decisions.

    Pass a nonempty tuple of stateless ``criteria``. To reproduce the review's
    combined beta rule, supply BetaBounds(consecutive=2) and
    BetaStability(consecutive=2). Each policy retains its sampling requirement;
    the final estimate must pass all of them.
    """

    def __init__(self, *, criteria: tuple[StoppingCriterion, ...]):
        self.criteria = tuple(criteria)
        if not self.criteria or not all(
            isinstance(criterion, StoppingCriterion) for criterion in self.criteria
        ):
            raise ValueError(
                "criteria must contain one or more StoppingCriterion objects"
            )

    def should_stop(self, history: tuple[LearningStep, ...]) -> bool:
        """Require all component policies on the same complete fit history."""
        return all(criterion.should_stop(history) for criterion in self.criteria)

    def accepts_estimate(self, estimate: ReliabilityEstimate) -> bool:
        """Require all component sampling checks without recomputing uncertainty."""
        return all(criterion.accepts_estimate(estimate) for criterion in self.criteria)
