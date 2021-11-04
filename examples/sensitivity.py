#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 23:09:16 2021

@author: ccaprani

This is Example 2 of Bourinet (2017), which gives analytical solutions as:
    
    \frac{\partial\beta}{\partial\mu_R} = 0.5184
    \frac{\partial\beta}{\partial\sigma_R} = −0.2548
    \frac{\partial\beta}{\partial\mu_S} = −1.3629
    \frac{\partial\beta}{\partial\sigma_S} = 0.0445

    Bourinet (2017), FORM Sensitivities to Distribution Parameters with the 
    Nataf Transformation, P. Gardoni (ed.), Risk and Reliability Analysis: 
        Theory and Applications, Springer Series in Reliability Engineering, 
        DOI 10.1007/978-3-319-52425-2_12

"""

import pystra as pr


def lsf(R, S):
    """
    Basic R-S
    """
    return R - S


limit_state = pr.LimitState(lsf)

model = pr.StochasticModel()
model.addVariable(pr.Lognormal("R", 5, 5))
model.addVariable(pr.Lognormal("S", 1, 1))
model.setCorrelation(pr.CorrelationMatrix([[1.0, 0.5], [0.5, 1.0]]))

form = pr.Form(stochastic_model=model, limit_state=limit_state)
form.showDetailedOutput()

sens = pr.SensitivityAnalysis(stochastic_model=model, limit_state=limit_state)
results = sens.run_form()
print(results)
