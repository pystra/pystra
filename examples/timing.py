#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

"""
This file examines the speed up possible when using the built-in distribution
function implementations, rather than the generic (but more comprehensive) ones
provided by SciPy. Where performance is critical, implementation of a 
user-defined distribution object that avoids error checks and user input 
validation on each call could be beneficial.

Called as: `$python timing.py` results in output like:
    
    ```
    Total time taken (s)
    Built-in: 1.8919757150579244; Scipy: 5.2964311360847205
    Average time per call (s):
    Built-in: 0.018919757150579242; Scipy: 0.052964311360847206
    Built-in speed-up: 2.80
    ```

"""
import pystra as ra
from scipy.stats import norm, lognorm, uniform
import numpy as np
import timeit


def lsf(r, X1, X2, X3):
    g = r - X2 / (1000 * X3) - (X1 / (200 * X3)) ** 2
    return g


def run_builtin():
    """
    The basic example using built-in distributions
    """
    limit_state = ra.LimitState(lsf)

    stochastic_model = ra.StochasticModel()
    stochastic_model.addVariable(ra.Lognormal("X1", 500, 100))
    stochastic_model.addVariable(ra.Normal("X2", 2000, 400))
    stochastic_model.addVariable(ra.Uniform("X3", 5, 0.5))
    stochastic_model.addVariable(ra.Constant("r", 1))

    stochastic_model.setCorrelation(
        ra.CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
    )

    sorm = ra.Sorm(
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    sorm.run()


def run_scipy():
    """
    The basic example using Scipy distributions
    """
    limit_state = ra.LimitState(lsf)

    stochastic_model = ra.StochasticModel()
    # Lognormal
    zeta = (np.log(1 + (100 / 500) ** 2)) ** 0.5
    lamb = np.log(500) - 0.5 * zeta ** 2
    stochastic_model.addVariable(
        ra.ScipyDist("X1", lognorm(s=zeta, scale=np.exp(lamb)))
    )
    # Normal
    stochastic_model.addVariable(ra.ScipyDist("X2", norm(loc=2000, scale=400)))
    ## Uniform
    a_b = (0.5 ** 2 * 12) ** (1 / 2)
    a = (2 * 5 - a_b) / 2
    stochastic_model.addVariable(ra.ScipyDist("X3", uniform(loc=a, scale=a_b)))
    # Constant
    stochastic_model.addVariable(ra.Constant("r", 1))

    stochastic_model.setCorrelation(
        ra.CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
    )

    sorm = ra.Sorm(
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    sorm.run()


number = 100
time_builtin = timeit.timeit(stmt=run_builtin, number=number)
time_scipy = timeit.timeit(stmt=run_scipy, number=number)

print("Total time taken (s)")
print(f"Built-in: {time_builtin}; Scipy: {time_scipy}")
print("Average time per call (s):")
print(f"Built-in: {time_builtin/number}; Scipy: {time_scipy/number}")
print(f"Built-in speed-up: {time_scipy/time_builtin:.2f}")
