#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:50:35 2021

@author: ccaprani
"""

import pystra as pr

from scipy.stats import genextreme as gev


def lsf(X1, X2, C):
    """
    Basic R-S
    """
    return X1 - X2 - C


# Create GEV variable and plot it is correct
X2 = pr.ScipyDist("X2", gev(c=0.1, loc=200, scale=50))
X2.plot()

limit_state = pr.LimitState(lsf)

model = pr.StochasticModel()
model.addVariable(pr.Normal("X1", 500, 100))
model.addVariable(X2)
model.addVariable(pr.Constant("C", 50))

form = pr.Form(stochastic_model=model, limit_state=limit_state)
form.showDetailedOutput()

sorm = pr.Sorm(stochastic_model=model, limit_state=limit_state, form=form)
sorm.run()
sorm.showDetailedOutput()
