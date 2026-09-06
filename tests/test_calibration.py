#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:40:01 2022

@author: shihab
"""

import pytest
import pystra as ra
import numpy as np
import pandas as pd


def lsf(z, R, G, Q1, Q2, cg, c1, c2):
    return z * R - (cg * G + c1 * Q1 + c2 * Q2)


def lsf3(z, R, G, Q1, Q2, Q3, cg, c1, c2, c3):
    return z * R - (cg * G + c1 * Q1 + c2 * Q2 + c3 * Q3)


def lsf_nonlinear(z, wR, wS, R, Q1, Q2):
    gX = z * wR * R - wS * (Q1 + Q2)
    return gX


def setup1():
    """
    Set up simulation for two varying load calibration problem
    Ref: Example 1, Caprani and Khan, Structural Safety, 2023
    """
    ## Define distributions of loads for combinations
    # Annual max distributions
    Q1max = ra.Gumbel("Q1", 1, 0.2)  # Imposed Load
    Q2max = ra.Gumbel("Q2", 1, 0.4)  # Wind Load
    # Parameters of inferred point-in-time parents
    Q1pit = ra.Gumbel("Q1", 0.89, 0.2)  # Imposed Load
    Q2pit = ra.Gumbel("Q2", 0.77, 0.4)  # Wind Load
    Q_dict = {"Q1": {"max": Q1max, "pit": Q1pit}, "Q2": {"max": Q2max, "pit": Q2pit}}
    # Constant values
    eg = ra.Constant("cg", 0.4)
    e1 = ra.Constant("c1", 0.6)
    e2 = ra.Constant("c2", 0.3)
    z = ra.Constant(
        "z", 1
    )  # Design parameter for resistance with arbitrary default value

    ## Define other random variables
    Rdist = ra.Lognormal("R", 1.0, 0.15)  # Resistance
    Gdist = ra.Normal("G", 1, 0.1)  # Permanent Load (static)

    loadcombinations = {"Q1_max": ["Q1"], "Q2_max": ["Q2"]}

    lc = ra.LoadCombination.from_actions(
        limit_state=lsf,
        maxima={name: values["max"] for name, values in Q_dict.items()},
        companions={name: values["pit"] for name, values in Q_dict.items()},
        resistance=[Rdist],
        other=[Gdist],
        constants=[z, eg, e1, e2],
        leading_actions=loadcombinations,
    )

    Qk = np.array([Q1max.ppf(0.98), Q2max.ppf(0.98)])
    Gk = np.array([Gdist.mean])
    Rk = np.array([Rdist.ppf(0.05)])
    rvs_all = ["R", "G", "Q1", "Q2", "Q3"]
    nominal_values = dict(zip(rvs_all, np.concatenate([Rk, Gk, Qk])))
    betaT = 4.3
    return lc, nominal_values, betaT


def setup2():
    """
    Set up simulation
    """
    ## Define distributions of loads for combinations
    # Annual max distributions
    Q1_max = ra.Normal("Q1", 30, 3)  # [units]
    Q2_max = ra.Normal("Q2", 20, 2)  # [units]

    z = ra.Constant("z", 1)
    # Parameters of arbitrary point-in-time parents
    Q1_pit = ra.Normal("Q1", 15, 3)  # [units]
    Q2_pit = ra.Normal("Q2", 10, 2)  # [units]
    Q_dict = {
        "Q1": {"max": Q1_max, "pit": Q1_pit},
        "Q2": {"max": Q2_max, "pit": Q2_pit},
    }
    # Constant values
    z = ra.Constant("z", 1)
    # Design parameter for resistance with arbitrary default value

    ## Define other random variables
    wR = ra.Lognormal("wR", 1.0, 0.05)
    wS = ra.Lognormal("wS", 1.0, 0.10)
    R = ra.Normal("R", 60, 6)  # [units]

    loadcombinations = {"Q1_max": ["Q1"], "Q2_max": ["Q2"]}

    lc = ra.LoadCombination.from_actions(
        limit_state=lsf_nonlinear,
        maxima={name: values["max"] for name, values in Q_dict.items()},
        companions={name: values["pit"] for name, values in Q_dict.items()},
        other=[wS],
        resistance=[R, wR],
        constants=[z],
        leading_actions=loadcombinations,
    )

    rvs_all = ["wR", "wS", "R", "Q1", "Q2"]
    nominal_values = dict(
        zip(
            rvs_all,
            np.array([1.0, 1.0, R.ppf(0.05), Q1_max.ppf(0.95), Q2_max.ppf(0.95)]),
        )
    )

    betaT = 3.7
    return lc, nominal_values, betaT


def setup3():
    """
    Set up simulation for three varying load calibration problem
    Ref: Example 2, Caprani and Khan, Structural Safety, 2023
    """
    ## Define distributions of loads for combinations
    # Annual max distributions
    Q1max = ra.Gumbel("Q1", 1, 0.2)
    Q2max = ra.Gumbel("Q2", 1, 0.3)
    Q3max = ra.Gumbel("Q3", 1, 0.4)
    # Parameters of inferred point-in-time parents
    Q1pit = ra.Gumbel("Q1", 0.887, 0.183)
    Q2pit = ra.Gumbel("Q2", 0.828, 0.278)
    Q3pit = ra.Gumbel("Q3", 0.802, 0.416)
    Q_dict = {
        "Q1": {"max": Q1max, "pit": Q1pit},
        "Q2": {"max": Q2max, "pit": Q2pit},
        "Q3": {"max": Q3max, "pit": Q3pit},
    }
    # Constant values
    cg = ra.Constant("cg", 0.2)
    c1 = ra.Constant("c1", 0.6)
    c2 = ra.Constant("c2", 0.35)
    c3 = ra.Constant("c3", 0.25)
    z = ra.Constant(
        "z", 1
    )  # Design parameter for resistance with arbitrary default value

    ## Define other random variables
    R = ra.Lognormal("R", 1.0, 0.15)
    G = ra.Normal("G", 1, 0.1)

    loadcombinations = {"Q1_max": ["Q1"], "Q2_max": ["Q2"], "Q3_max": ["Q3"]}
    lc = ra.LoadCombination.from_actions(
        limit_state=lsf3,
        maxima={name: values["max"] for name, values in Q_dict.items()},
        companions={name: values["pit"] for name, values in Q_dict.items()},
        resistance=[R],
        other=[G],
        constants=[z, cg, c1, c2, c3],
        leading_actions=loadcombinations,
    )

    Qk = np.array([Q1max.ppf(0.95), Q2max.ppf(0.95), Q3max.ppf(0.90)])
    Gk = np.array([G.ppf(0.5)])
    Rk = np.array([R.ppf(0.05)])
    rvs_all = ["R", "G", "Q1", "Q2", "Q3"]
    nominal_values = dict(zip(rvs_all, np.concatenate([Rk, Gk, Qk])))
    betaT = 4.8
    return lc, nominal_values, betaT


def test_calibration_coeff_opt():
    """
    Perform SORM analysis
    """
    lc, nominal_values, betaT = setup1()
    problem = ra.FactorCalibrationProblem(
        lc, nominal_values=nominal_values, design_parameter="z"
    )
    solved = ra.solve_designs(problem, target_beta=betaT, method="root")
    assert solved.converged
    factors = ra.derive_factors(solved, method="coeff")
    design_points = pd.DataFrame(
        data=[
            [0.6553, 1.0371, 1.6236, 2.0171, 3.0431],
            [0.6550, 1.0371, 1.5129, 2.2458, 3.0477],
        ],
        columns=["R", "G", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    resistance_factors = pd.DataFrame(
        data=[[0.8469], [0.8465]], columns=["R"], index=["Q1_max", "Q2_max"]
    )
    load_factors = pd.DataFrame(
        data=[[1.0371, 1.0692, 1.1026], [1.0371, 1.0692, 1.1026]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    combination_factors = pd.DataFrame(
        data=[[1.0, 1.0, 0.8982], [1.0, 0.9318, 1.0]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z1 = np.array([3.0443, 3.0477])
    vect_design_beta1 = np.array([4.3065, 4.3000])
    # validate results
    assert pytest.approx(solved.to_frame(), abs=1e-4) == design_points
    assert pytest.approx(factors.to_frame("resistance"), abs=1e-4) == resistance_factors
    assert pytest.approx(factors.to_frame("loads"), abs=1e-4) == load_factors
    assert (
        pytest.approx(factors.to_frame("combinations"), abs=1e-4) == combination_factors
    )
    assert (
        pytest.approx(
            ra.design_with_factors(
                problem,
                ra.select_factors(
                    factors,
                    resistance="minimum",
                    loads="maximum",
                    combinations="maximum",
                ),
            ).values,
            abs=1e-4,
        )
        == vect_design_z1
    )
    assert (
        pytest.approx(
            np.array(
                [
                    check.reliability.beta
                    for check in ra.verify_designs(problem, np.max(vect_design_z1))
                ]
            ),
            abs=1e-4,
        )
        == vect_design_beta1
    )


def test_calibration_mat_opt():
    """
    Perform SORM analysis
    """
    lc, nominal_values, betaT = setup1()
    problem = ra.FactorCalibrationProblem(
        lc, nominal_values=nominal_values, design_parameter="z"
    )
    solved = ra.solve_designs(problem, target_beta=betaT, method="root")
    assert solved.converged
    factors = ra.derive_factors(solved, method="matrix")
    design_points = pd.DataFrame(
        data=[
            [0.6553, 1.0371, 1.6236, 2.0171, 3.0431],
            [0.6550, 1.0371, 1.5129, 2.2458, 3.0477],
        ],
        columns=["R", "G", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    resistance_factors = pd.DataFrame(
        data=[[0.8469], [0.8465]], columns=["R"], index=["Q1_max", "Q2_max"]
    )
    load_factors = pd.DataFrame(
        data=[[1.0371, 1.0692, 1.1026], [1.0371, 1.0692, 1.1026]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    combination_factors = pd.DataFrame(
        data=[[1.0, 1.0, 0.8982], [1.0, 0.9318, 1.0]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z2 = np.array([3.0443, 3.0477])
    vect_design_beta2 = np.array([4.3065, 4.3000])
    # validate results
    assert pytest.approx(solved.to_frame(), abs=1e-4) == design_points
    assert pytest.approx(factors.to_frame("resistance"), abs=1e-4) == resistance_factors
    assert pytest.approx(factors.to_frame("loads"), abs=1e-4) == load_factors
    assert (
        pytest.approx(factors.to_frame("combinations"), abs=1e-4) == combination_factors
    )
    assert (
        pytest.approx(
            ra.design_with_factors(
                problem,
                ra.select_factors(
                    factors,
                    resistance="minimum",
                    loads="maximum",
                    combinations="maximum",
                ),
            ).values,
            abs=1e-4,
        )
        == vect_design_z2
    )
    assert (
        pytest.approx(
            np.array(
                [
                    check.reliability.beta
                    for check in ra.verify_designs(problem, np.max(vect_design_z2))
                ]
            ),
            abs=1e-4,
        )
        == vect_design_beta2
    )


def test_calibration_mat_alpha():
    """
    Perform SORM analysis
    """
    lc, nominal_values, betaT = setup1()
    problem = ra.FactorCalibrationProblem(
        lc, nominal_values=nominal_values, design_parameter="z"
    )
    solved = ra.solve_designs(problem, target_beta=betaT, method="alpha")
    assert solved.converged
    factors = ra.derive_factors(solved, method="matrix")
    design_points = pd.DataFrame(
        data=[
            [0.6553, 1.0371, 1.6236, 2.0171, 3.0431],
            [0.6550, 1.0371, 1.5129, 2.2458, 3.0477],
        ],
        columns=["R", "G", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    resistance_factors = pd.DataFrame(
        data=[[0.8469], [0.8465]], columns=["R"], index=["Q1_max", "Q2_max"]
    )
    load_factors = pd.DataFrame(
        data=[[1.0371, 1.0692, 1.1026], [1.0371, 1.0692, 1.1026]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    combination_factors = pd.DataFrame(
        data=[[1.0, 1.0, 0.8982], [1.0, 0.9318, 1.0]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z3 = np.array([3.0443, 3.0477])
    vect_design_beta3 = np.array([4.3065, 4.3000])
    # validate results
    assert pytest.approx(solved.to_frame(), abs=1e-4) == design_points
    assert pytest.approx(factors.to_frame("resistance"), abs=1e-4) == resistance_factors
    assert pytest.approx(factors.to_frame("loads"), abs=1e-4) == load_factors
    assert (
        pytest.approx(factors.to_frame("combinations"), abs=1e-4) == combination_factors
    )
    assert (
        pytest.approx(
            ra.design_with_factors(
                problem,
                ra.select_factors(
                    factors,
                    resistance="minimum",
                    loads="maximum",
                    combinations="maximum",
                ),
            ).values,
            abs=1e-4,
        )
        == vect_design_z3
    )
    assert (
        pytest.approx(
            np.array(
                [
                    check.reliability.beta
                    for check in ra.verify_designs(problem, np.max(vect_design_z3))
                ]
            ),
            abs=1e-4,
        )
        == vect_design_beta3
    )


def test_calibration_coeff_opt_nonlinear():
    """
    Perform SORM analysis
    """
    lc, nominal_values, betaT = setup2()
    problem = ra.FactorCalibrationProblem(
        lc, nominal_values=nominal_values, design_parameter="z"
    )
    solved = ra.solve_designs(problem, target_beta=betaT, method="root")
    assert solved.converged
    factors = ra.derive_factors(solved, method="coeff")
    design_points = pd.DataFrame(
        data=[
            [44.4005, 0.9519, 1.2050, 33.8055, 11.6913, 1.2971],
            [44.7632, 0.9526, 1.2014, 19.1578, 21.8479, 1.1553],
        ],
        columns=["R", "wR", "wS", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    resistance_factors = pd.DataFrame(
        data=[[0.8857, 0.9519], [0.8929, 0.9526]],
        columns=["R", "wR"],
        index=["Q1_max", "Q2_max"],
    )
    load_factors = pd.DataFrame(
        data=[[1.2050, 0.9677, 0.9381], [1.2014, 0.9677, 0.9381]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    combination_factors = pd.DataFrame(
        data=[[1.0, 1.0, 0.5351], [1.0, 0.5667, 1.0]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z1 = np.array([1.2971, 1.1587])
    vect_design_beta1 = np.array([3.7001, 4.2835])
    # validate results
    assert pytest.approx(solved.to_frame(), abs=1e-4) == design_points
    assert pytest.approx(factors.to_frame("resistance"), abs=1e-4) == resistance_factors
    assert pytest.approx(factors.to_frame("loads"), abs=1e-4) == load_factors
    assert (
        pytest.approx(factors.to_frame("combinations"), abs=1e-4) == combination_factors
    )
    assert (
        pytest.approx(
            ra.design_with_factors(
                problem, ra.select_factors(factors, loads="maximum")
            ).values,
            abs=1e-3,
        )
        == vect_design_z1
    )
    assert (
        pytest.approx(
            np.array(
                [
                    check.reliability.beta
                    for check in ra.verify_designs(problem, np.max(vect_design_z1))
                ]
            ),
            abs=1e-3,
        )
        == vect_design_beta1
    )


def test_calibration_mat_opt_nonlinear():
    """
    Perform SORM analysis
    """
    lc, nominal_values, betaT = setup2()
    problem = ra.FactorCalibrationProblem(
        lc, nominal_values=nominal_values, design_parameter="z"
    )
    solved = ra.solve_designs(problem, target_beta=betaT, method="root")
    assert solved.converged
    factors = ra.derive_factors(solved, method="matrix")
    design_points = pd.DataFrame(
        data=[
            [44.4005, 0.9519, 1.2050, 33.8055, 11.6913, 1.2971],
            [44.7632, 0.9526, 1.2014, 19.1578, 21.8479, 1.1553],
        ],
        columns=["R", "wR", "wS", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    resistance_factors = pd.DataFrame(
        data=[[0.8857, 0.9519], [0.8929, 0.9526]],
        columns=["R", "wR"],
        index=["Q1_max", "Q2_max"],
    )
    load_factors = pd.DataFrame(
        data=[[1.2050, 0.9677, 0.9381], [1.2014, 0.9677, 0.9381]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    combination_factors = pd.DataFrame(
        data=[[1.0, 1.0, 0.5367], [1.0, 0.5651, 1.0]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z2 = np.array([1.2980, 1.1571])
    vect_design_beta2 = np.array([3.7037, 4.2869])
    # validate results
    assert pytest.approx(solved.to_frame(), abs=1e-4) == design_points
    assert pytest.approx(factors.to_frame("resistance"), abs=1e-4) == resistance_factors
    assert pytest.approx(factors.to_frame("loads"), abs=1e-4) == load_factors
    assert (
        pytest.approx(factors.to_frame("combinations"), abs=1e-4) == combination_factors
    )
    assert (
        pytest.approx(
            ra.design_with_factors(
                problem, ra.select_factors(factors, loads="maximum")
            ).values,
            abs=1e-3,
        )
        == vect_design_z2
    )
    assert (
        pytest.approx(
            np.array(
                [
                    check.reliability.beta
                    for check in ra.verify_designs(problem, np.max(vect_design_z2))
                ]
            ),
            abs=1e-3,
        )
        == vect_design_beta2
    )


def test_calibration_mat_alpha_nonlinear():
    """
    Perform SORM analysis
    """
    lc, nominal_values, betaT = setup2()
    problem = ra.FactorCalibrationProblem(
        lc, nominal_values=nominal_values, design_parameter="z"
    )
    solved = ra.solve_designs(problem, target_beta=betaT, method="alpha")
    assert solved.converged
    factors = ra.derive_factors(solved, method="matrix")
    design_points = pd.DataFrame(
        data=[
            [44.4005, 0.9519, 1.2050, 33.8055, 11.6913, 1.2971],
            [44.7632, 0.9526, 1.2014, 19.1578, 21.8479, 1.1553],
        ],
        columns=["R", "wR", "wS", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    resistance_factors = pd.DataFrame(
        data=[[0.8857, 0.9519], [0.8929, 0.9526]],
        columns=["R", "wR"],
        index=["Q1_max", "Q2_max"],
    )
    load_factors = pd.DataFrame(
        data=[[1.2050, 0.9677, 0.9381], [1.2014, 0.9677, 0.9381]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    combination_factors = pd.DataFrame(
        data=[[1.0, 1.0, 0.5367], [1.0, 0.5651, 1.0]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z3 = np.array([1.2980, 1.1571])
    vect_design_beta3 = np.array([3.7037, 4.2869])
    # validate results
    assert pytest.approx(solved.to_frame(), abs=1e-4) == design_points
    assert pytest.approx(factors.to_frame("resistance"), abs=1e-4) == resistance_factors
    assert pytest.approx(factors.to_frame("loads"), abs=1e-4) == load_factors
    assert (
        pytest.approx(factors.to_frame("combinations"), abs=1e-4) == combination_factors
    )
    assert (
        pytest.approx(
            ra.design_with_factors(
                problem, ra.select_factors(factors, loads="maximum")
            ).values,
            abs=1e-4,
        )
        == vect_design_z3
    )
    assert (
        pytest.approx(
            np.array(
                [
                    check.reliability.beta
                    for check in ra.verify_designs(problem, np.max(vect_design_z3))
                ]
            ),
            abs=1e-3,
        )
        == vect_design_beta3
    )


def test_calibration_coeff_opt_3():
    """
    Perform SORM analysis
    """
    lc, nominal_values, betaT = setup3()
    problem = ra.FactorCalibrationProblem(
        lc, nominal_values=nominal_values, design_parameter="z"
    )
    solved = ra.solve_designs(problem, target_beta=betaT, method="root")
    assert solved.converged
    factors = ra.derive_factors(solved, method="coeff")
    design_points = pd.DataFrame(
        data=[
            [0.6194, 1.0194, 1.8722, 1.2591, 1.6108, 3.5045],
            [0.6137, 1.0202, 1.4497, 1.727, 1.7667, 3.4546],
            [0.6124, 1.0207, 1.5489, 1.3686, 1.8671, 3.3951],
        ],
        columns=["R", "G", "Q1", "Q2", "Q3", "z"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    resistance_factors = pd.DataFrame(
        data=[[0.8005], [0.7931], [0.7915]],
        columns=["R"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    load_factors = pd.DataFrame(
        data=[
            [1.0194, 1.3634, 1.1072, 1.2269],
            [1.0202, 1.3634, 1.1072, 1.2269],
            [1.0207, 1.3634, 1.1072, 1.2269],
        ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    combination_factors = pd.DataFrame(
        data=[
            [1.0, 1.0, 0.7291, 0.8627],
            [1.0, 0.7743, 1.0, 0.9463],
            [1.0, 0.8273, 0.7925, 1.0],
        ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    # print(ra.design_with_factors(problem, ra.select_factors(factors, resistance="minimum", loads="maximum", combinations="maximum")).values)
    vect_design_z1 = np.array([3.6709, 3.559, 3.3951])
    vect_design_beta1 = np.array([5.0028, 5.0708, 5.1493])
    # validate results
    assert pytest.approx(solved.to_frame(), abs=1e-4) == design_points
    assert pytest.approx(factors.to_frame("resistance"), abs=1e-4) == resistance_factors
    assert pytest.approx(factors.to_frame("loads"), abs=1e-4) == load_factors
    assert (
        pytest.approx(factors.to_frame("combinations"), abs=1e-4) == combination_factors
    )
    assert (
        pytest.approx(
            ra.design_with_factors(
                problem,
                ra.select_factors(
                    factors,
                    resistance="minimum",
                    loads="maximum",
                    combinations="maximum",
                ),
            ).values,
            abs=1e-4,
        )
        == vect_design_z1
    )
    assert (
        pytest.approx(
            np.array(
                [
                    check.reliability.beta
                    for check in ra.verify_designs(problem, np.max(vect_design_z1))
                ]
            ),
            abs=1e-4,
        )
        == vect_design_beta1
    )


def test_calibration_mat_opt_3():
    """
    Perform SORM analysis
    """
    lc, nominal_values, betaT = setup3()
    problem = ra.FactorCalibrationProblem(
        lc, nominal_values=nominal_values, design_parameter="z"
    )
    solved = ra.solve_designs(problem, target_beta=betaT, method="root")
    assert solved.converged
    factors = ra.derive_factors(solved, method="matrix")
    design_points = pd.DataFrame(
        data=[
            [0.6194, 1.0194, 1.8722, 1.2591, 1.6108, 3.5045],
            [0.6137, 1.0202, 1.4497, 1.7270, 1.7667, 3.4546],
            [0.6124, 1.0207, 1.5489, 1.3686, 1.8671, 3.3951],
        ],
        columns=["R", "G", "Q1", "Q2", "Q3", "z"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    resistance_factors = pd.DataFrame(
        data=[[0.8005], [0.7931], [0.7915]],
        columns=["R"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    load_factors = pd.DataFrame(
        data=[
            [1.0194, 1.3634, 1.1072, 1.2269],
            [1.0202, 1.3634, 1.1072, 1.2269],
            [1.0207, 1.3634, 1.1072, 1.2269],
        ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    combination_factors = pd.DataFrame(
        data=[
            [1.0, 1.0, 0.7778, 0.7997],
            [1.0, 0.8352, 1.0, 0.7997],
            [1.0, 0.8352, 0.7778, 1.0],
        ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    vect_design_z2 = np.array([3.5442, 3.4616, 3.3951])
    vect_design_beta2 = np.array([4.8494, 4.9144, 4.9925])
    # print(ra.design_with_factors(problem, ra.select_factors(factors, resistance="minimum", loads="maximum", combinations="maximum")).values)
    # validate results
    assert pytest.approx(solved.to_frame(), abs=1e-4) == design_points
    assert pytest.approx(factors.to_frame("resistance"), abs=1e-4) == resistance_factors
    assert pytest.approx(factors.to_frame("loads"), abs=1e-4) == load_factors
    assert (
        pytest.approx(factors.to_frame("combinations"), abs=1e-4) == combination_factors
    )
    assert (
        pytest.approx(
            ra.design_with_factors(
                problem,
                ra.select_factors(
                    factors,
                    resistance="minimum",
                    loads="maximum",
                    combinations="maximum",
                ),
            ).values,
            abs=1e-3,
        )
        == vect_design_z2
    )
    assert (
        pytest.approx(
            np.array(
                [
                    check.reliability.beta
                    for check in ra.verify_designs(problem, np.max(vect_design_z2))
                ]
            ),
            abs=1e-4,
        )
        == vect_design_beta2
    )


def test_calibration_mat_alpha_3():
    """
    Perform SORM analysis
    """
    lc, nominal_values, betaT = setup3()
    problem = ra.FactorCalibrationProblem(
        lc, nominal_values=nominal_values, design_parameter="z"
    )
    solved = ra.solve_designs(problem, target_beta=betaT, method="alpha")
    assert solved.converged
    factors = ra.derive_factors(solved, method="matrix")
    design_points = pd.DataFrame(
        data=[
            [0.6194, 1.0194, 1.8722, 1.2591, 1.6108, 3.5045],
            [0.6137, 1.0202, 1.4497, 1.7270, 1.7667, 3.4546],
            [0.6124, 1.0207, 1.5489, 1.3686, 1.8671, 3.3951],
        ],
        columns=["R", "G", "Q1", "Q2", "Q3", "z"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    resistance_factors = pd.DataFrame(
        data=[[0.8005], [0.7931], [0.7915]],
        columns=["R"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    load_factors = pd.DataFrame(
        data=[
            [1.0194, 1.3634, 1.1072, 1.2269],
            [1.0202, 1.3634, 1.1072, 1.2269],
            [1.0207, 1.3634, 1.1072, 1.2269],
        ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    combination_factors = pd.DataFrame(
        data=[
            [1.0, 1.0, 0.7778, 0.7997],
            [1.0, 0.8352, 1.0, 0.7997],
            [1.0, 0.8352, 0.7778, 1.0],
        ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    vect_design_z3 = np.array([3.5442, 3.4616, 3.3951])
    vect_design_beta3 = np.array([4.8494, 4.9144, 4.9925])
    # validate results
    assert pytest.approx(solved.to_frame(), abs=1e-4) == design_points
    assert pytest.approx(factors.to_frame("resistance"), abs=1e-4) == resistance_factors
    assert pytest.approx(factors.to_frame("loads"), abs=1e-4) == load_factors
    assert (
        pytest.approx(factors.to_frame("combinations"), abs=1e-4) == combination_factors
    )
    assert (
        pytest.approx(
            ra.design_with_factors(
                problem,
                ra.select_factors(
                    factors,
                    resistance="minimum",
                    loads="maximum",
                    combinations="maximum",
                ),
            ).values,
            abs=1e-3,
        )
        == vect_design_z3
    )
    assert (
        pytest.approx(
            np.array(
                [
                    check.reliability.beta
                    for check in ra.verify_designs(problem, np.max(vect_design_z3))
                ]
            ),
            abs=1e-4,
        )
        == vect_design_beta3
    )
