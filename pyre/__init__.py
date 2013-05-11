"""
Markov Chain Monte Carlo sampling toolkit.

Bayesian estimation, particularly using Markov chain Monte Carlo (MCMC), is an increasingly relevant approach to statistical estimation. However, few statistical software packages implement MCMC samplers, and they are non-trivial to code by hand. pymc is a python package that implements the Metropolis-Hastings algorithm as a python class, and is extremely flexible and applicable to a large suite of problems. pymc includes methods for summarizing output, plotting, goodness-of-fit and convergence diagnostics.

pymc only requires NumPy. All other dependencies such as matplotlib, SciPy, pytables, or sqlite are optional.

"""

__version__ = '2.2'


try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy does not seem to be installed. Please see the user guide.')

# __modules__ = ['distribution',
#                'normal']

# from .distribution import *
# from .normal import *

# Distributions
from distributions import *
from correlation import *

# Inputparameter
from model import *
from function import *

# Calculations
from limitstate import *
from cholesky import *
from stepsize import *
from quadrature import *
from integration import *

# Transformation
from transformation import *

# Analysis
from form import *
