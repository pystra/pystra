"""Immutable records separating surrogate diagnostics and sampling error."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ._validation import _positive_integer

__all__ = ["ReliabilityEstimate", "LearningStep", "ActiveLearningResult"]


@dataclass(frozen=True)
class ReliabilityEstimate:
    """Conditional failure estimate returned by a reliability estimator.

    ``sampling_cov`` quantifies sampling error conditional on the fitted
    surrogate; it excludes surrogate bias. Infinity denotes unresolved
    precision. ``sampling_interval`` is either None or a (lower, upper) pair
    at ``confidence_level``. Its calculation belongs to the estimator, as do
    importance weights and corrections for dependent samples. ``method``
    names that calculation; ``sampling_dependence`` is 'independent' or
    'dependent'. ``n_samples`` counts surrogate classifications, not true
    limit-state evaluations.

    ``converged`` means the estimator reached its target event, separately
    from satisfying a requested CoV. ``status`` explains incomplete sampling.
    ``diagnostics`` retains immutable method-specific records (SubsetRun for
    replicated subset simulation). Incomplete estimates must not be accepted
    as successful reliability results.
    """

    failure_probability: float
    sampling_cov: float
    sampling_interval: tuple | None
    confidence_level: float | None
    n_samples: int
    method: str
    sampling_dependence: str
    converged: bool = True
    status: str = "completed"
    diagnostics: tuple = ()

    def __post_init__(self):
        if (
            not np.isfinite(self.failure_probability)
            or not 0 <= self.failure_probability <= 1
        ):
            raise ValueError("failure_probability must be finite and in [0, 1]")
        if np.isnan(self.sampling_cov) or self.sampling_cov < 0:
            raise ValueError("sampling_cov must be nonnegative or infinity")
        _positive_integer(self.n_samples, "n_samples", 2)
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must describe the sampling calculation")
        if self.sampling_dependence not in ("independent", "dependent"):
            raise ValueError("sampling_dependence must be 'independent' or 'dependent'")
        if not isinstance(self.converged, bool):
            raise ValueError("converged must be a bool")
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.sampling_interval is None:
            if self.confidence_level is not None:
                raise ValueError("confidence_level requires a sampling_interval")
        else:
            interval = tuple(float(value) for value in self.sampling_interval)
            if len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 1:
                raise ValueError("sampling_interval must be an ordered pair in [0, 1]")
            if self.confidence_level is None or not 0 < self.confidence_level < 1:
                raise ValueError("confidence_level must be in (0, 1)")
            object.__setattr__(self, "sampling_interval", interval)

    @property
    def beta(self) -> float:
        """Normal-equivalent index; infinite at probability zero or one."""
        return float(-norm.ppf(self.failure_probability))


@dataclass(frozen=True)
class LearningStep:
    """One fit's probability diagnostics and cumulative true evaluation count.

    ``probability_band`` is the ascending pair of probabilities for
    mean + 2*std <= 0 and mean - 2*std <= 0. The fixed MC workflow uses
    candidate proportions; adaptive estimators supply their own measure.
    ``beta_band`` transforms those
    endpoints into ascending reliability indices. These are surrogate
    sensitivity diagnostics, not confidence intervals or true Pf bounds.
    ``learning_satisfied`` is the selection policy's threshold flag, before
    applying the stopping policy. The default requires an interior exploratory
    failure probability, under the estimator's sampling measure.
    ``estimation_converged`` and optional ``sampling_cov`` record the
    exploratory estimator's completion and precision, not the final sample.
    ``bootstrap_probability_band`` is the min/max of actual replicate Pf
    estimates on fixed IID normal enrichment, when requested. It need not
    enclose the full-design mean prediction and is not a confidence interval.
    """

    failure_probability: float
    learning_score: float
    n_limit_state_evaluations: int
    probability_band: tuple
    beta_band: tuple
    learning_satisfied: bool
    estimation_converged: bool = True
    sampling_cov: float | None = None
    bootstrap_probability_band: tuple | None = None

    def __post_init__(self):
        if self.bootstrap_probability_band is not None:
            band = tuple(float(value) for value in self.bootstrap_probability_band)
            if len(band) != 2 or not 0 <= band[0] <= band[1] <= 1:
                raise ValueError("bootstrap_probability_band must be ordered in [0, 1]")
            object.__setattr__(self, "bootstrap_probability_band", band)

    @property
    def beta(self) -> float:
        """Normal-equivalent index of the candidate probability."""
        return float(-norm.ppf(self.failure_probability))


@dataclass(frozen=True)
class ActiveLearningResult:
    """Snapshot including unfinished estimates, diagnostics and stopping status.

    ``estimate`` identifies the final estimator and its uncertainty semantics.
    Probability, beta, sampling diagnostics and n_estimation are exposed as
    convenient properties of that record. ``converged`` means the configured
    stopping and sampling criteria passed; it does not guarantee accuracy or
    discovery of all failure regions. History contains exploratory probability
    diagnostics, separate from the independent final estimate.
    """

    estimate: ReliabilityEstimate
    converged: bool
    status: str
    n_limit_state_evaluations: int
    history: tuple

    @property
    def failure_probability(self) -> float:
        """Final estimator's conditional failure probability."""
        return self.estimate.failure_probability

    @property
    def beta(self) -> float:
        """Normal-equivalent index of the final probability."""
        return self.estimate.beta

    @property
    def sampling_cov(self) -> float:
        """Final estimator's conditional sampling coefficient of variation."""
        return self.estimate.sampling_cov

    @property
    def sampling_interval(self) -> tuple | None:
        """Final estimator's conditional sampling interval, if available."""
        return self.estimate.sampling_interval

    @property
    def n_estimation(self) -> int:
        """Number of surrogate classifications used by the final estimator."""
        return self.estimate.n_samples
