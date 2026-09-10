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

# Distributions
from .distributions import *
from .dependence.correlation import CorrelationMatrix
from .dependence.copula import *
from .dependence.joint import *

# Inputparameter
from .model import *

# Analysis
from .reliability.analysis import *
from .results import FORMResult
from .reliability.form import *
from .reliability.monte_carlo import *
from .reliability.sorm import *
from .reliability.line_sampling import *
from .reliability.subset_simulation import *
from .reliability.sensitivity import *
from .systems import *
from .reliability.system_form import *
from .reliability.strong_maximum import *

# Calibration
from .fbc import *
from .loadcomb import *
from .calibration import *

# Design decision optimization
from .decision import ddo
from .decision.ddo import (
    CostBenefitModel,
    DDO,
    DDOCriterion,
    DDOObjective,
    DesignStudy,
    FatalityConsequence,
    LQI,
    RackwitzTargetModel,
    RiskResult,
    RiskStudy,
    ScenarioRiskModel,
    SWTP,
    TargetReliability,
)

# Figure helpers accept existing axes and never display figures implicitly.
from . import plotting
