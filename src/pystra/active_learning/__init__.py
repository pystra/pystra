"""Composable active-learning methods for structural reliability.

The four components are a surrogate, reliability estimator, learning function
and stopping criterion; see Moustapha, Marelli and Sudret (2022),
doi:10.1016/j.strusafe.2021.102174, and Teixeira, Nogal and O'Connor (2021),
doi:10.1016/j.strusafe.2020.102019. Current enrichment uses a fixed normal MC
pool; the estimator interface controls the independent final estimation.
Kriging requires the optional ``al`` extra. PCE uses only NumPy/SciPy.
"""

from .analysis import ActiveLearning
from .results import ActiveLearningResult, LearningStep, ReliabilityEstimate
from .surrogates import (
    Surrogate,
    KrigingSurrogate,
    PceSurrogate,
    PceFitResult,
    PceCandidate,
)
from .learning import (
    LearningFunction,
    LearningDecision,
    UFunction,
    ExpectedFeasibility,
    learning_u,
    learning_eff,
)
from .estimation import ReliabilityEstimator, MonteCarloEstimator
from .stopping import (
    StoppingCriterion,
    LearningThreshold,
    BetaBounds,
    BetaStability,
    AllCriteria,
)

__all__ = [
    "ActiveLearning",
    "ActiveLearningResult",
    "LearningStep",
    "ReliabilityEstimate",
    "Surrogate",
    "KrigingSurrogate",
    "PceSurrogate",
    "PceFitResult",
    "PceCandidate",
    "LearningFunction",
    "LearningDecision",
    "UFunction",
    "ExpectedFeasibility",
    "learning_u",
    "learning_eff",
    "ReliabilityEstimator",
    "MonteCarloEstimator",
    "StoppingCriterion",
    "LearningThreshold",
    "BetaBounds",
    "BetaStability",
    "AllCriteria",
]
