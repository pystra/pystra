#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

"""
This file examplifies the system reliability module. From Melhem Thesis.

"""
import pystra as pr
# from scipy.stats import norm, lognorm, uniform
import numpy as np
# import timeit

# Define and create limit state function (the same for all components) 
def lsf(r, s):
    return r - s

ls = pr.LimitState(lsf)

# Create components

comp1 = pr.Component('A',ls)
comp2 = pr.Component('B',ls)
comp3 = pr.Component('C',ls)

comp_list = [comp1,comp2,comp3]

# Define and name random variables network as seen in limit state functions

## Example parameters
R1 = 100        # Mean resistance in A
R2 = 200        # Mean resistance in B
R3 = 150        # Mean resistance in C
S1 = 90         # Mean loading in A
S2 = 190        # Mean loading in B
S3 = 140        # Mean loading in C

# create random variables (using bias and cov trick)
rv_1 = pr.Normal('r', *R1*1.00*np.array([1, 0.1]))
rv_4 = pr.Normal('s', *S1*1.00*np.array([1, 5/90]))
rv_2 = pr.Normal('r', *R2*1.00*np.array([1, 0.1]))
rv_5 = pr.Normal('s', *S2*1.00*np.array([1, 6/190]))
rv_3 = pr.Normal('r', *R3*1.00*np.array([1, 0.1]))
rv_6 = pr.Normal('s', *S3*1.00*np.array([1, 4/140]))

# Add random variables to components
comp1.addVariable(rv_1)
comp1.addVariable(rv_4)
comp2.addVariable(rv_2)
comp2.addVariable(rv_5)
comp3.addVariable(rv_3)
comp3.addVariable(rv_6)

# See betas for interests 
comp_betas = []
for c in comp_list:
    comp_betas.append(c.getProbability()[0])

# Now create system(s)

series_sys = pr.SeriesSystem(comp_list)
parallel_sys = pr.ParallelSystem(comp_list)
serpar_sys = pr.SeriesSystem([comp1,
                              pr.ParallelSystem([comp2,comp3])])

# For interests, get event probabilities (should be the same since same components!)
# default is equivalent planes method

series_sys.getEventProbabilities()
parallel_sys.getEventProbabilities()
serpar_sys.getEventProbabilities()


# calculating system beta and pf

series_reliability = series_sys.getReliability()
parallel_reliability = parallel_sys.getReliability()
serpar_reliability = serpar_sys.getReliability()
