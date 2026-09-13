import pystra as ra
import numpy as np


def lsf(z, R, G, Q1, Q2, cg):
    return z * R - (cg * G + 0.6 * Q1 + 0.3 * Q2)


Q1max = ra.Gumbel("Q1", 1, 0.2)  # Imposed Load
Q2max = ra.Gumbel("Q2", 1, 0.4)  # Wind Load

Q1pit = ra.Gumbel("Q1", 0.89, 0.2)  # Imposed Load
Q2pit = ra.Gumbel("Q2", 0.77, 0.4)  # Wind Load
Q_dict = {"Q1": {"max": Q1max, "pit": Q1pit}, "Q2": {"max": Q2max, "pit": Q2pit}}

cg = ra.Constant("cg", 0.4)
z = ra.Constant("z", 1)  # Design parameter for resistance with arbitrary default value

Rdist = ra.Lognormal("R", 1.0, 0.15)  # Resistance
Gdist = ra.Normal("G", 1, 0.1)  # Permanent Load (other load variable)

loadcombinations = {"Q1_max": ["Q1"], "Q2_max": ["Q2"]}

lc = ra.LoadCombination.from_actions(
    limit_state=lsf,
    maxima={name: distributions["max"] for name, distributions in Q_dict.items()},
    companions={name: distributions["pit"] for name, distributions in Q_dict.items()},
    resistance=[Rdist],
    other=[Gdist],
    constants=[z, cg],
    leading_actions=loadcombinations,
)

Qk = np.array([Q1max.ppf(0.98), Q2max.ppf(0.98)])
Gk = np.array([Gdist.ppf(0.5)])
Rk = np.array([Rdist.ppf(0.05)])
rvs_all = ["R", "G", "Q1", "Q2", "Q3"]
dict_nom = dict(zip(rvs_all, np.concatenate([Rk, Gk, Qk])))

betaT = 4.3

ra.StdNormal.cdf(-betaT)
trial_calibration = []

problem = ra.calibration.FactorCalibrationProblem(
    lc, nominal_values=dict_nom, design_parameter="z"
)
solutions = ra.calibration.solve_designs(problem, target_beta=betaT, method="root")
assert solutions.converged
factors = ra.calibration.derive_factors(solutions, method="coeff")
selected = ra.calibration.select_factors(
    factors, resistance="minimum", loads="maximum", combinations="maximum"
)
designs = ra.calibration.design_with_factors(problem, selected)
verified = ra.calibration.verify_designs(
    problem, max(designs.values), target_beta=betaT
)
trial_calibration.append(
    {
        "designs": list(designs.values),
        "beta": [row.reliability.beta for row in verified],
    }
)
print(solutions.to_frame())

design_z = np.array(designs.values)
design_beta = np.array([row.reliability.beta for row in verified])
print(f"Design reliabilities = {design_beta.round(2)}")
print(f"Design Check = {design_beta.round(2)>=betaT}")

problem = ra.calibration.FactorCalibrationProblem(
    lc, nominal_values=dict_nom, design_parameter="z"
)
solutions = ra.calibration.solve_designs(problem, target_beta=betaT, method="root")
assert solutions.converged
factors = ra.calibration.derive_factors(solutions, method="matrix")
selected = ra.calibration.select_factors(
    factors, resistance="minimum", loads="maximum", combinations="maximum"
)
designs = ra.calibration.design_with_factors(problem, selected)
verified = ra.calibration.verify_designs(
    problem, max(designs.values), target_beta=betaT
)
trial_calibration.append(
    {
        "designs": list(designs.values),
        "beta": [row.reliability.beta for row in verified],
    }
)
print(solutions.to_frame())

design_z = np.array(designs.values)
design_beta = np.array([row.reliability.beta for row in verified])
print(f"Design reliabilities = {design_beta.round(2)}")
print(f"Design Check = {design_beta.round(2)>=betaT}")

problem = ra.calibration.FactorCalibrationProblem(
    lc, nominal_values=dict_nom, design_parameter="z"
)
solutions = ra.calibration.solve_designs(problem, target_beta=betaT, method="alpha")
assert solutions.converged
factors = ra.calibration.derive_factors(solutions, method="matrix")
selected = ra.calibration.select_factors(
    factors, resistance="minimum", loads="maximum", combinations="maximum"
)
designs = ra.calibration.design_with_factors(problem, selected)
verified = ra.calibration.verify_designs(
    problem, max(designs.values), target_beta=betaT
)
trial_calibration.append(
    {
        "designs": list(designs.values),
        "beta": [row.reliability.beta for row in verified],
    }
)
print(solutions.to_frame())

design_z = np.array(designs.values)
design_beta = np.array([row.reliability.beta for row in verified])
print(f"Design reliabilities = {design_beta.round(2)}")
print(f"Design Check = {design_beta.round(2)>=betaT}")

import pystra as ra
import numpy as np


def lsf(z, wR, wS, R, Q1, Q2):
    gX = z * wR * R - wS * (Q1 + Q2)
    return gX


wR = ra.Lognormal("wR", 1.0, 0.05)
wS = ra.Lognormal("wS", 1.0, 0.10)
R = ra.Normal("R", 60, 6)  # [units]
Q1_max = ra.Normal("Q1", 30, 3)  # [units]
Q2_max = ra.Normal("Q2", 20, 2)  # [units]
Q1_pit = ra.Normal("Q1", 15, 3)  # [units]
Q2_pit = ra.Normal("Q2", 10, 2)  # [units]

z = ra.Constant("z", 1)

rvs_all = ["wR", "wS", "R", "Q1", "Q2"]
dict_nom = dict(
    zip(rvs_all, np.array([1.0, 1.0, R.ppf(0.05), Q1_max.ppf(0.95), Q2_max.ppf(0.95)]))
)

Q_dict = {"Q1": {"max": Q1_max, "pit": Q1_pit}, "Q2": {"max": Q2_max, "pit": Q2_pit}}

loadcombinations = {"Q1_max": ["Q1"], "Q2_max": ["Q2"]}

lc = ra.LoadCombination.from_actions(
    limit_state=lsf,
    maxima={name: distributions["max"] for name, distributions in Q_dict.items()},
    companions={name: distributions["pit"] for name, distributions in Q_dict.items()},
    resistance=[R, wR],
    other=[wS],
    constants=[z],
    leading_actions=loadcombinations,
)

betaT = 3.7
problem = ra.calibration.FactorCalibrationProblem(
    lc, nominal_values=dict_nom, design_parameter="z"
)
solutions = ra.calibration.solve_designs(problem, target_beta=betaT, method="root")
assert solutions.converged
factors = ra.calibration.derive_factors(solutions, method="matrix")
selected = ra.calibration.select_factors(
    factors, resistance="minimum", loads="maximum", combinations="maximum"
)
designs = ra.calibration.design_with_factors(problem, selected)
verified = ra.calibration.verify_designs(
    problem, max(designs.values), target_beta=betaT
)
trial_calibration.append(
    {
        "designs": list(designs.values),
        "beta": [row.reliability.beta for row in verified],
    }
)

print(solutions.to_frame())

design_z1 = np.array(designs.values)
design_beta1 = np.array([row.reliability.beta for row in verified])
print(f"Design reliabilities = {design_beta1.round(2)}")
print(f"Design Check = {design_beta1.round(2)>=betaT}")
