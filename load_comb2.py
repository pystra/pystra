#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load Combinations - the Ferry Borges-Castanheta model

Thoft-Christensen & Baker - Structural Reliability Theory and Its Applications
Example 10.6

Created on Wed Sep 22 02:57:44 2021

@author: ccaprani
"""

import numpy as np
import pyre as pr
import matplotlib.pyplot as plt


def g(Mf, P1, P2, P3):
    return Mf - 25 * (P1 + P2 + P3) / 8


def load_comb(n1, n2, n3):
    limit_state = pr.LimitState(g)

    options = pr.AnalysisOptions()
    options.setE1(1e-3)  # Design point accuracy
    options.setE2(5e-3)  # gradient vector direction accuracy
    options.setImax(350)  # Additional iters for difficult LSF

    # Establish model
    prob_model = pr.StochasticModel()

    # Define random variables
    prob_model.addVariable(pr.Normal("Mf", 12.5, 1.25))

    p1 = pr.Maximum("P1", pr.Normal("P1", 0.50, 0.2), n1)
    # p1 = pr.NormalN("P1", 0.50, 0.2, n1)
    # p1.setStartPoint(2.0)
    prob_model.addVariable(p1)

    p2 = pr.Maximum("P2", pr.Normal("P2", -0.2, 0.4), n2)
    # p2 = pr.NormalN("P2", -0.2, 0.4, n2)
    # p2.setStartPoint(0.8)
    prob_model.addVariable(p2)

    p3 = pr.Maximum("P3", pr.Normal("P3", -2.00, 1.0), n3)
    # p3 = pr.NormalN("P3", -2.00, 1.0, n3)
    # p3.setStartPoint(1.0)
    prob_model.addVariable(p3)

    # Perform FORM
    form = pr.Form(
        stochastic_model=prob_model, limit_state=limit_state, analysis_options=options
    )
    form.showDetailedOutput()

    return form.getBeta()


load_comb(n1=1, n2=12, n3=360)


def plot_pdfs():
    x = np.linspace(-1, 2, 200)
    p = pr.Lognormal("P1", mean=0.50, stdv=0.2)
    n = 10
    pdf = [pr.Lognormal.pdf(xi, p.getP1(), p.getP2()) for xi in x]
    pdfn = [pr.NormalN.pdf(xi, p.getMean(), p.getStdv(), n) for xi in x]
    pdfn2 = [pr.Maximum.pdf(xi, p, n) for xi in x]

    plt.plot(x, pdf, label="Parent")
    plt.plot(x, pdfn, "r-", label="NormalN")
    plt.plot(x, pdfn2, "go:", label="Maximum")
    plt.legend()
