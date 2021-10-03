#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load Combinations - the Ferry Borges-Castanheta model

Thoft-Christensen & Baker - Structural Reliability Theory and Its Applications
Example 10.2 & 10.3

Created on Tue Sep 21 22:26:45 2021

@author: ccaprani
"""

import numpy as np
import pyre as pr
import matplotlib.pyplot as plt


def g(Mf, P1, P2):
    return Mf - 5 / 2 * (P1 + P2)


def load_repeats(n):
    limit_state = pr.LimitState(g)

    # Establish model
    prob_model = pr.StochasticModel()

    # Define random variables
    prob_model.addVariable(pr.Normal("Mf", 20, 2))
    prob_model.addVariable(pr.Normal("P1", 3, 0.3))
    # prob_model.addVariable(pr.NormalN("P2", 2, 0.2, n))
    prob_model.addVariable(pr.Maximum("P2", pr.Normal("P2parent", 2, 0.2), n))

    # Perform FORM
    form = pr.Form(stochastic_model=prob_model, limit_state=limit_state)
    form.showDetailedOutput()

    return form.getBeta()


# Figure 10.8 of book
beta = [load_repeats(n) for n in range(1, 21)]
plt.plot(beta)
