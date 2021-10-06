#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 11:15:14 2021

@author: ccaprani
"""

import pyre as pr

import time
import datetime


def example_limitstatefunction(X1, X2, X3):
    """
    example limit state function
    """
    return 1 - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2


start_time = time.time()

limit_state = pr.LimitState(example_limitstatefunction)

stochastic_model = pr.StochasticModel()
# Define random variables
stochastic_model.addVariable(pr.Lognormal("X1", 500, 100))
stochastic_model.addVariable(pr.Normal("X2", 2000, 400))
stochastic_model.addVariable(pr.Uniform("X3", 5, 0.5))

# If the random variables are correlatet, then define a correlation matrix,
# else no correlatin matrix is needed
stochastic_model.setCorrelation(
    pr.CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
)

# Perform FORM analysis
Analysis = pr.Form(stochastic_model=stochastic_model, limit_state=limit_state)
# More detailed output
Analysis.showDetailedOutput()

sorm = pr.Sorm(
    stochastic_model=stochastic_model, limit_state=limit_state, form=Analysis
)
sorm.run()
# Detailed output
sorm.showDetailedOutput()

# Perform Distribution analysis
Analysis = pr.DistributionAnalysis(
    stochastic_model=stochastic_model, limit_state=limit_state
)


run_time = time.time() - start_time
print(f"{1e3*run_time:.3f} ms")
