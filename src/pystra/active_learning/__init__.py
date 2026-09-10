"""Composable active-learning methods for structural reliability.

The four components are a surrogate, reliability estimator, learning function
and stopping criterion; see Moustapha, Marelli and Sudret (2022),
doi:10.1016/j.strusafe.2021.102174, and Teixeira, Nogal and O'Connor (2021),
doi:10.1016/j.strusafe.2020.102019. Enrichment uses a fixed normal MC pool by
default; SubsetSimulationEstimator generates adaptive conditional populations
with its own probability diagnostics and independent final estimation.
Kriging requires the optional ``al`` extra. PCE uses only NumPy/SciPy.
"""

from .analysis import ActiveLearning
from .results import ActiveLearningResult, LearningStep, ReliabilityEstimate
from .surrogates import (
    Surrogate,
    EnsembleSurrogate,
    KrigingSurrogate,
    PCESurrogate,
    PCEFitResult,
    PCECandidate,
)
from .learning import (
    LearningFunction,
    EnsembleLearningFunction,
    FBRLearning,
    LearningDecision,
    UFunction,
    ExpectedFeasibility,
    learning_u,
    learning_eff,
)
from .estimation import (
    ReliabilityEstimator,
    MonteCarloEstimator,
    EnrichmentEstimator,
    EnrichmentResult,
)
from .pc_kriging import PCKrigingSurrogate, PCKrigingFitResult
from .importance import ImportanceSamplingEstimator, ImportanceSamplingDiagnostics
from .subset import SubsetSimulationEstimator, SubsetRun, SubsetLevel
from .stopping import (
    StoppingCriterion,
    BootstrapBounds,
    LearningThreshold,
    BetaBounds,
    BetaStability,
    AllCriteria,
)

__all__ = [
    "EnsembleSurrogate",
    "EnsembleLearningFunction",
    "FBRLearning",
    "BootstrapBounds",
    "PCKrigingSurrogate",
    "PCKrigingFitResult",
    "ImportanceSamplingEstimator",
    "ImportanceSamplingDiagnostics",
    "ActiveLearning",
    "ActiveLearningResult",
    "LearningStep",
    "ReliabilityEstimate",
    "Surrogate",
    "KrigingSurrogate",
    "PCESurrogate",
    "PCEFitResult",
    "PCECandidate",
    "LearningFunction",
    "LearningDecision",
    "UFunction",
    "ExpectedFeasibility",
    "learning_u",
    "learning_eff",
    "ReliabilityEstimator",
    "MonteCarloEstimator",
    "EnrichmentEstimator",
    "EnrichmentResult",
    "SubsetSimulationEstimator",
    "SubsetRun",
    "SubsetLevel",
    "StoppingCriterion",
    "LearningThreshold",
    "BetaBounds",
    "BetaStability",
    "AllCriteria",
]
