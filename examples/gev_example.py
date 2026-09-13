#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze a resistance-load model with a SciPy GEV marginal.

Author: ccaprani.
"""

from scipy.stats import genextreme as gev

import pystra as ra


def lsf(X1, X2, C):
    """
    Basic R-S
    """
    return X1 - X2 - C


# Create GEV variable and plot it is correct
X2 = ra.ScipyDist("X2", gev(c=0.1, loc=200, scale=50))
X2.plot()

limit_state = ra.LimitState(lsf)

model = ra.StochasticModel()
model.add_variable(ra.Normal("X1", 500, 100))
model.add_variable(X2)
model.add_variable(ra.Constant("C", 50))

form = ra.FORM(model=model, limit_state=limit_state)
form_result = form.run()
print(form_result.summary())

sorm = ra.SORM(model=model, limit_state=limit_state, form=form)
sorm_result = sorm.run()
print(sorm_result.summary())
