"""
Pystra — Python Structural Reliability Analysis.

Pystra provides tools for computing the probability of failure of
engineering systems using established structural reliability methods:

- **FORM** (First Order Reliability Method)
- **SORM** (Second Order Reliability Method)
- **Monte Carlo** simulation (Crude, Line Sampling, Subset Simulation)
- **Sensitivity analysis** of the reliability index
- **Load combination** and **partial factor calibration**

All SciPy continuous distributions can be used as random variables,
alongside Pystra's own specialised distributions (e.g. Gumbel,
ZeroInflated, Maximum).

Quick start::

    import pystra as ra

    limit_state = ra.LimitState(lambda R, S: R - S)
    model = ra.StochasticModel()
    model.addVariable(ra.Normal("R", 10, 1))
    model.addVariable(ra.Normal("S", 5, 1))

    form = ra.Form(
        stochastic_model=model,
        limit_state=limit_state,
    )
    form.run()
    print(f"beta = {form.getBeta():.4f}")
"""

__version__ = "1.6.0"

# Distributions
from .distributions import *
from .correlation import *

# Inputparameter
from .model import *

# Calculations
from .quadrature import *
from .integration import *

# Transformation
from .transformation import *

# Analysis
from .analysis import *
from .form import *
from .mc import *
from .sorm import *
from .ls import *
from .ss import *
from .sensitivity import *

# Calibration
from .loadcomb import *
from .calibration import *
