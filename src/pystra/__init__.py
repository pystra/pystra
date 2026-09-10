"""
Pystra — Python Structural Reliability Analysis.

Pystra provides tools for computing the probability of failure of
engineering systems using established structural reliability methods:

- **FORM** (First Order Reliability Method)
- **SORM** (Second Order Reliability Method)
- **Monte Carlo** simulation (Crude, Line Sampling, Subset Simulation)
- **Sensitivity analysis** of the reliability index
- **Load combination** and **partial factor calibration**
- **Design decision optimization** with societal risk acceptance checks

All SciPy continuous distributions can be used as random variables,
alongside Pystra's own specialised distributions (e.g. Gumbel,
ZeroInflated, Maximum).

Quick start::

    import pystra as ra

    limit_state = ra.LimitState(lambda R, S: R - S)
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("R", 10, 1))
    model.add_variable(ra.Normal("S", 5, 1))

    form = ra.FORM(
        stochastic_model=model,
        limit_state=limit_state,
    )
    form.run()
    print(f"beta = {form.get_beta():.4f}")
"""

__version__ = "2.0.0.dev0"

from .distributions import (
    Distribution,
    StdNormal,
    Normal,
    Lognormal,
    Uniform,
    Beta,
    Gamma,
    ChiSquare,
    ShiftedExponential,
    ShiftedLognormal,
    ShiftedRayleigh,
    Gumbel,
    GumbelMin,
    Frechet,
    Weibull,
    GEV,
    GEVmax,
    GEVMin,
    Maximum,
    MaxParent,
    ZeroInflated,
    ScipyDist,
    Constant,
)
from .dependence import (
    CorrelationMatrix,
    JointDistribution,
    GaussianCopula,
    StudentTCopula,
    FrankCopula,
    IndependentCopula,
    Transformation,
)
from .model import (
    StochasticModel,
    LimitState,
)
from .reliability import (
    AnalysisObject,
    AnalysisOptions,
    FORM,
    SORM,
    MonteCarlo,
    CrudeMonteCarlo,
    ImportanceSampling,
    DistributionAnalysis,
    LineSampling,
    SubsetSimulation,
    SensitivityAnalysis,
    SystemFORM,
    StrongMaximumTest,
)
from .results import (
    FORMResult,
)
from .systems import (
    Component,
    System,
    SeriesSystem,
    ParallelSystem,
    CutSetSystem,
    TieSetSystem,
    KOfNSystem,
    ditlevsen_bounds,
)
from .loads import (
    FBCProcess,
    VariableRoles,
    LoadCombination,
)
from .active_learning import (
    ActiveLearning,
    ActiveLearningResult,
)
from . import calibration, decision, plotting
from ._signposts import TOP_LEVEL as _SIGNPOSTS

__all__ = [
    "Distribution",
    "StdNormal",
    "Normal",
    "Lognormal",
    "Uniform",
    "Beta",
    "Gamma",
    "ChiSquare",
    "ShiftedExponential",
    "ShiftedLognormal",
    "ShiftedRayleigh",
    "Gumbel",
    "GumbelMin",
    "Frechet",
    "Weibull",
    "GEV",
    "GEVmax",
    "GEVMin",
    "Maximum",
    "MaxParent",
    "ZeroInflated",
    "ScipyDist",
    "Constant",
    "CorrelationMatrix",
    "JointDistribution",
    "GaussianCopula",
    "StudentTCopula",
    "FrankCopula",
    "IndependentCopula",
    "Transformation",
    "StochasticModel",
    "LimitState",
    "AnalysisObject",
    "AnalysisOptions",
    "FORM",
    "SORM",
    "MonteCarlo",
    "CrudeMonteCarlo",
    "ImportanceSampling",
    "DistributionAnalysis",
    "LineSampling",
    "SubsetSimulation",
    "SensitivityAnalysis",
    "SystemFORM",
    "StrongMaximumTest",
    "FORMResult",
    "Component",
    "System",
    "SeriesSystem",
    "ParallelSystem",
    "CutSetSystem",
    "TieSetSystem",
    "KOfNSystem",
    "ditlevsen_bounds",
    "FBCProcess",
    "VariableRoles",
    "LoadCombination",
    "ActiveLearning",
    "ActiveLearningResult",
]

_SUBPACKAGES = (
    "calibration",
    "decision",
    "dependence",
    "reliability",
    "loads",
    "systems",
    "active_learning",
    "distributions",
    "plotting",
)


def __getattr__(name):
    """Name the replacement when code uses a 1.x name, a moved module or a subpackage name."""
    if name in _SIGNPOSTS:
        raise AttributeError(_SIGNPOSTS[name])
    if not name.startswith("__"):
        for sub in _SUBPACKAGES:
            if name in getattr(globals()[sub], "__all__", ()):
                raise AttributeError(
                    f"{name} is not exported from pystra; import it from pystra.{sub}"
                )
    raise AttributeError(f"module 'pystra' has no attribute {name!r}")
