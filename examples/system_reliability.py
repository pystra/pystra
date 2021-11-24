#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

"""
This file examplifies the system reliability module. 

"""
import pystra as pr
import numpy as np
import matplotlib.pyplot as plt

# making some options changes
options = pr.AnalysisOptions()
options.print_output = False

#%%
"""
Series system and parallel system 
correlated random variables in each limit state function 
same random variables in limit state
Problem 3 from Bora Gencturk - FERUM manual on MSR
"""

# LSF 1


def lsf_1(x1, x2):
    return 64 - (x1 + 5) ** 2 - x2 ** 2


ls_1 = pr.LimitState(lsf_1)
comp_1 = pr.Component("g1", ls_1, options)

# LSF 2


def lsf_2(x1, x2):
    return 64 - (x1) ** 2 - (x2 + 5) ** 2


ls_2 = pr.LimitState(lsf_2)
comp_2 = pr.Component("g2", ls_2, options)

# LSF 3


def lsf_3(x1, x2):
    return 64 - (x1 + 2) ** 2 - (x2 + 3) ** 2


ls_3 = pr.LimitState(lsf_3)
comp_3 = pr.Component("g3", ls_3, options)

# LSF 4


def lsf_4(x1, x2):
    return 64 - (x1 + 3) ** 2 - (x2 + 2) ** 2


ls_4 = pr.LimitState(lsf_4)
comp_4 = pr.Component("g4", ls_4, options)


comp_list = [comp_1, comp_2, comp_3, comp_4]

# define random variables and set component correlations

p = 0.2
# even if this set to zero, there is still correlation because shared variables
for c in comp_list:
    c.addVariable(pr.Normal("x1", 0, 2))
    c.addVariable(pr.Lognormal("x2", 1, 1))

    c.setCorrelation(pr.CorrelationMatrix([[1.0, p], [p, 1.0]]))

# Now create system(s)

series_sys = pr.SeriesSystem(comp_list)
parallel_sys = pr.ParallelSystem(comp_list)

systems = [series_sys, parallel_sys]

# Define autocorrelation (order as per random variables instances)
# descring correlation between the same variable

autocorr = [1, 1]
# full correlation (ie. the exact same variable), (this is default)

comp_betas = []
for c in comp_list:
    comp_betas.append(c.getProbability())

for s in systems:
    s.setCorrelation(np.array(autocorr))

# calculating system beta and pf

for s in systems:
    print(s.getReliability())

# matches with example!
# Series: 1.034 x 10-1
# Parallel: 7.677 x 10-3

#%%

"""
Series system and parallel system with correlated variables: 
Thoft-Christensen and Sorenson (1982)
"Reliability of structural systems with correlated elements"
"""

# Define and create limit state function (the same for all components)
def lsf(r, s):
    return r - s


# Here r and s can be the same as the component correlations is being provided

ls = pr.LimitState(lsf)

results = {}

# Create n components
for N in [2, 5, 10]:

    comp_list = []
    for n in range(0, N):
        comp_list.append(pr.Component("B " + str(n), ls, options))

    # Create variables such that component beta is the same
    component_beta = 3.0
    sd_R = 10  # Dummy Standard deviation resistance
    mu_R = 90  # Dummy mean resistance (same for all)
    S = mu_R - component_beta * sd_R  # dummy loading

    # create random variables and add to component
    for c in comp_list:
        c.addVariable(pr.Normal("r", mu_R, sd_R))
        c.addVariable(pr.Constant("s", S))

    # See betas for interests
    comp_betas = []
    for c in comp_list:
        comp_betas.append(c.getProbability()[0])

    # Now create system(s)

    series_sys = pr.SeriesSystem(comp_list)
    parallel_sys = pr.ParallelSystem(comp_list)

    systems = [series_sys, parallel_sys]

    # Define autocorrelation (order as per random variables instances)
    # descring correlation between the same variable

    pf_list = {"series": [], "parallel": []}

    for p in np.linspace(0, 1, 11):

        autocorr = [p]

        # Define equal correlation between components

        for s in systems:
            s.setCorrelation(np.array(autocorr))

        # calculating system beta and pf

        pf_list["series"].append([p, series_sys.getReliability()[-1]])
        pf_list["parallel"].append([p, parallel_sys.getReliability()[-1]])

    results[N] = {key: np.array(pf_list[key]) for key in pf_list.keys()}

# plot results

fig, ax = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
sys_dict = {"series": 0, "parallel": 1}

for n, v_dict in results.items():
    for sys, vals in v_dict.items():
        ax[sys_dict[sys]].plot(vals[:, 0], vals[:, 1], label="n = " + str(n))

for k, i in sys_dict.items():
    plt.xlabel(r"$\rho$")  # x_label
    ax[i].set_ylabel(r"$P_f$ - " + k + " system")  # y_label
    ax[i].legend()
    ax[i].grid()


plt.savefig("system_validation_plot.pdf")

# series system matches
# parallel system does not. (ductile system?)

#%%
"""
Series system with correlated random variables of same type: 
Example 1 from Zhou, Gong & Hong (2017)
"New Perspective on Application of First-Order Reliability
Method for Estimating System Reliability"
"""

# Nominal values
(Dn, tn, SMTS, Po, ln) = (610, 7.16, 517, 6.0, 50)  # mm and MPa
d_mu = np.array([0.25, 0.30]) * tn  # for defects 1 and 2

# Define and create limit state function (the same for all components)
def lsf(D, t, s, p, l, d, x):

    geo_term = l ** 2 / (D * t)

    # case for less than 50 (since geo_term is nominal)
    M = np.sqrt(1 + 0.6275 * geo_term - 0.003375 * geo_term ** 2)

    pb = x * (1.8 * t * s) / D * ((1 - d / t) / (1 - d / (M * t)))

    # lsf for a pipeline defect
    g = pb - p

    return g


ls = pr.LimitState(lsf)

# make components
comp_list = []
for n in range(0, 2):
    comp_list.append(pr.Component("B " + str(n + 1), ls, options))

# add random variables (no correlation within component)
for i, c in enumerate(comp_list):
    c.addVariable(pr.Constant("D", Dn))
    c.addVariable(pr.Constant("t", tn))
    c.addVariable(pr.Lognormal("s", 1.09 * SMTS, 0.03 * 1.09 * SMTS))
    c.addVariable(pr.Gumbel("p", 1.05 * Po, 0.1 * 1.05 * Po))
    c.addVariable(pr.Constant("l", ln))
    c.addVariable(pr.Weibull("d", d_mu[i], 0.2 * d_mu[i]))
    c.addVariable(pr.Lognormal("x", 1.10, 0.172 * 1.10))

# check component betas
comp_betas = []
for c in comp_list:
    comp_betas.append(c.getProbability())

# Now create system(s)
series_sys = pr.SeriesSystem(comp_list)

# Define autocorrelation (order as per random variables instances)
# descring correlation between the same random variables

autocorr = [0.3, 0.8, 0.5, 0.5]  # in order of s,p,d,x

series_sys.setCorrelation(np.array(autocorr))

# calculating system beta and pf

print(series_sys.getReliability())

# Results from paper
# β1 3.25, β2 3.19, and ρ12 0.64 
# the failure probability 8.2 × 10−4
# not matching...hmmm...