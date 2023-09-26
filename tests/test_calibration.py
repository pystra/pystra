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

    lc = ra.LoadCombination(
        lsf=lsf,
        dict_dist_comb=Q_dict,
        list_dist_resist=[Rdist],
        list_dist_other=[Gdist],
        list_const=[z, eg, e1, e2],
        dict_comb_cases=loadcombinations,
    )

    Qk = np.array([Q1max.ppf(0.98), Q2max.ppf(0.98)])
    Gk = np.array([Gdist.mean])
    Rk = np.array([Rdist.ppf(0.05)])
    rvs_all = ["R", "G", "Q1", "Q2", "Q3"]
    dict_nom = dict(zip(rvs_all, np.concatenate([Rk, Gk, Qk])))
    betaT = 4.3
    return lc, dict_nom, betaT


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

    lc = ra.LoadCombination(
        lsf=lsf_nonlinear,
        dict_dist_comb=Q_dict,
        list_dist_other=[wS],
        list_dist_resist=[R, wR],
        list_const=[z],
        dict_comb_cases=loadcombinations,
    )

    rvs_all = ["wR", "wS", "R", "Q1", "Q2"]
    dict_nom = dict(
        zip(
            rvs_all,
            np.array([1.0, 1.0, R.ppf(0.05), Q1_max.ppf(0.95), Q2_max.ppf(0.95)]),
        )
    )

    betaT = 3.7
    return lc, dict_nom, betaT

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
    Q_dict = {"Q1": {"max": Q1max, "pit": Q1pit},
              "Q2": {"max": Q2max, "pit": Q2pit},
              "Q3": {"max": Q3max, "pit": Q3pit}}
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

    loadcombinations = {'Q1_max':['Q1'], 'Q2_max':['Q2'], 'Q3_max':['Q3']}
    lc = ra.LoadCombination(lsf=lsf3, dict_dist_comb=Q_dict, 
                            list_dist_resist=[R], 
                            list_dist_other=[G], 
                            list_const=[z, cg, c1, c2, c3],
                            dict_comb_cases=loadcombinations)

    Qk = np.array([Q1max.ppf(0.95), Q2max.ppf(0.95), Q3max.ppf(0.90)])
    Gk = np.array([G.ppf(0.5)])
    Rk = np.array([R.ppf(0.05)])
    rvs_all = ['R', 'G', 'Q1', 'Q2', 'Q3']
    dict_nom = dict(zip(rvs_all, np.concatenate([Rk, Gk, Qk])))
    betaT = 4.8
    return lc, dict_nom, betaT

def test_calibration_coeff_opt():
    """
    Perform SORM analysis
    """
    lc, dict_nom, betaT = setup1()
    calib1 = ra.Calibration(
        lc,
        target_beta=betaT,
        dict_nom_vals=dict_nom,
        calib_var="z",
        est_method="coeff",
        calib_method="optimize",
        print_output=False,
    )
    calib1.run()
    dfXst = pd.DataFrame(
        data=[
            [0.6553, 1.0371, 1.6236, 2.0171, 3.0431],
            [0.6550, 1.0371, 1.5129, 2.2458, 3.0477],
        ],
        columns=["R", "G", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    dfphi = pd.DataFrame(
        data=[[0.8469], [0.8465]], columns=["R"], index=["Q1_max", "Q2_max"]
    )
    dfgamma = pd.DataFrame(
        data=[[1.0371, 1.0692, 1.1026], [1.0371, 1.0692, 1.1026]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    dfpsi = pd.DataFrame(
        data=[[1.0, 1.0, 0.8982], [1.0, 0.9318, 1.0]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z1 = np.array([3.04313479, 3.04771357])
    vect_design_beta1 = np.array([4.30646625, 4.30000037])
    # validate results
    assert pytest.approx(calib1.dfXstarcal, abs=1e-4) == dfXst
    assert pytest.approx(calib1.df_phi, abs=1e-4) == dfphi
    assert pytest.approx(calib1.df_gamma, abs=1e-4) == dfgamma
    assert pytest.approx(calib1.df_psi, abs=1e-4) == dfpsi
    assert pytest.approx(calib1.get_design_param_factor(), abs=1e-4) == vect_design_z1
    assert (
        pytest.approx(calib1.calc_beta_design_param(np.max(vect_design_z1)), abs=1e-4)
        == vect_design_beta1
    )


def test_calibration_mat_opt():
    """
    Perform SORM analysis
    """
    lc, dict_nom, betaT = setup1()
    calib2 = ra.Calibration(
        lc,
        target_beta=betaT,
        dict_nom_vals=dict_nom,
        calib_var="z",
        est_method="matrix",
        calib_method="optimize",
        print_output=False,
    )
    calib2.run()
    dfXst = pd.DataFrame(
        data=[
            [0.6553, 1.0371, 1.6236, 2.0171, 3.0431],
            [0.6550, 1.0371, 1.5129, 2.2458, 3.0477],
        ],
        columns=["R", "G", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    dfphi = pd.DataFrame(
        data=[[0.8469], [0.8465]], columns=["R"], index=["Q1_max", "Q2_max"]
    )
    dfgamma = pd.DataFrame(
        data=[[1.0371, 1.0692, 1.1026], [1.0371, 1.0692, 1.1026]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    dfpsi = pd.DataFrame(
        data=[[1.0, 1.0, 0.8982], [1.0, 0.9318, 1.0]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z2 = np.array([3.04313453, 3.04771331])
    vect_design_beta2 = np.array([4.30646588, 4.3])
    # validate results
    assert pytest.approx(calib2.dfXstarcal, abs=1e-4) == dfXst
    assert pytest.approx(calib2.df_phi, abs=1e-4) == dfphi
    assert pytest.approx(calib2.df_gamma, abs=1e-4) == dfgamma
    assert pytest.approx(calib2.df_psi, abs=1e-4) == dfpsi
    assert pytest.approx(calib2.get_design_param_factor(), abs=1e-4) == vect_design_z2
    assert (
        pytest.approx(calib2.calc_beta_design_param(np.max(vect_design_z2)), abs=1e-4)
        == vect_design_beta2
    )


def test_calibration_mat_alpha():
    """
    Perform SORM analysis
    """
    lc, dict_nom, betaT = setup1()
    calib3 = ra.Calibration(
        lc,
        target_beta=betaT,
        dict_nom_vals=dict_nom,
        calib_var="z",
        est_method="matrix",
        calib_method="alpha",
        print_output=False,
    )
    calib3.run()
    dfXst = pd.DataFrame(
        data=[
            [0.6553, 1.0371, 1.6236, 2.0171, 3.0431],
            [0.6550, 1.0371, 1.5129, 2.2458, 3.0477],
        ],
        columns=["R", "G", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    dfphi = pd.DataFrame(
        data=[[0.8469], [0.8465]], columns=["R"], index=["Q1_max", "Q2_max"]
    )
    dfgamma = pd.DataFrame(
        data=[[1.0371, 1.0692, 1.1026], [1.0371, 1.0692, 1.1026]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    dfpsi = pd.DataFrame(
        data=[[1.0, 1.0, 0.8982], [1.0, 0.9318, 1.0]],
        columns=["G", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z3 = np.array([3.04313111, 3.04770901])
    vect_design_beta3 = np.array([4.30645983, 4.29999394])
    # validate results
    assert pytest.approx(calib3.dfXstarcal, abs=1e-4) == dfXst
    assert pytest.approx(calib3.df_phi, abs=1e-4) == dfphi
    assert pytest.approx(calib3.df_gamma, abs=1e-4) == dfgamma
    assert pytest.approx(calib3.df_psi, abs=1e-4) == dfpsi
    assert pytest.approx(calib3.get_design_param_factor(), abs=1e-4) == vect_design_z3
    assert (
        pytest.approx(calib3.calc_beta_design_param(np.max(vect_design_z3)), abs=1e-4)
        == vect_design_beta3
    )


def test_calibration_coeff_opt_nonlinear():
    """
    Perform SORM analysis
    """
    lc, dict_nom, betaT = setup2()
    calib1 = ra.Calibration(
        lc,
        target_beta=betaT,
        dict_nom_vals=dict_nom,
        calib_var="z",
        est_method="coeff",
        calib_method="optimize",
    )
    calib1.run()
    dfXst = pd.DataFrame(
        data=[
            [44.4005, 0.9519, 1.2050, 33.8055, 11.6913, 1.2971],
            [44.7632, 0.9526, 1.2014, 19.1578, 21.8479, 1.1553],
        ],
        columns=["R", "wR", "wS", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    dfphi = pd.DataFrame(
        data=[[0.8857, 0.9519], [0.8929, 0.9526]],
        columns=["R", "wR"],
        index=["Q1_max", "Q2_max"],
    )
    dfgamma = pd.DataFrame(
        data=[[1.2050, 0.9677, 0.9381], [1.2014, 0.9677, 0.9381]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    dfpsi = pd.DataFrame(
        data=[[1.0, 1.0, 0.5351], [1.0, 0.5667, 1.0]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z1 = np.array([1.2971, 1.1553])
    vect_design_beta1 = np.array([3.7001, 4.2834])
    # validate results
    assert pytest.approx(calib1.dfXstarcal, abs=1e-4) == dfXst
    assert pytest.approx(calib1.df_phi, abs=1e-4) == dfphi
    assert pytest.approx(calib1.df_gamma, abs=1e-4) == dfgamma
    assert pytest.approx(calib1.df_psi, abs=1e-4) == dfpsi
    assert pytest.approx(calib1.get_design_param_factor(), abs=1e-4) == vect_design_z1
    assert (
        pytest.approx(calib1.calc_beta_design_param(np.max(vect_design_z1)), abs=1e-3)
        == vect_design_beta1
    )


def test_calibration_mat_opt_nonlinear():
    """
    Perform SORM analysis
    """
    lc, dict_nom, betaT = setup2()
    calib2 = ra.Calibration(
        lc,
        target_beta=betaT,
        dict_nom_vals=dict_nom,
        calib_var="z",
        est_method="matrix",
        calib_method="optimize",
        print_output=False,
    )
    calib2.run()
    dfXst = pd.DataFrame(
        data=[
            [44.4005, 0.9519, 1.2050, 33.8055, 11.6913, 1.2971],
            [44.7632, 0.9526, 1.2014, 19.1578, 21.8479, 1.1553],
        ],
        columns=["R", "wR", "wS", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    dfphi = pd.DataFrame(
        data=[[0.8857, 0.9519], [0.8929, 0.9526]],
        columns=["R", "wR"],
        index=["Q1_max", "Q2_max"],
    )
    dfgamma = pd.DataFrame(
        data=[[1.2050, 0.9677, 0.9381], [1.2014, 0.9677, 0.9381]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    dfpsi = pd.DataFrame(
        data=[[1.0, 1.0, 0.5367], [1.0, 0.5651, 1.0]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z2 = np.array([1.2980, 1.1537])
    vect_design_beta2 = np.array([3.7037, 4.2869])
    # validate results
    assert pytest.approx(calib2.dfXstarcal, abs=1e-4) == dfXst
    assert pytest.approx(calib2.df_phi, abs=1e-4) == dfphi
    assert pytest.approx(calib2.df_gamma, abs=1e-4) == dfgamma
    assert pytest.approx(calib2.df_psi, abs=1e-4) == dfpsi
    assert pytest.approx(calib2.get_design_param_factor(), abs=1e-4) == vect_design_z2
    assert (
        pytest.approx(calib2.calc_beta_design_param(np.max(vect_design_z2)), abs=1e-3)
        == vect_design_beta2
    )


def test_calibration_mat_alpha_nonlinear():
    """
    Perform SORM analysis
    """
    lc, dict_nom, betaT = setup2()
    calib3 = ra.Calibration(
        lc,
        target_beta=betaT,
        dict_nom_vals=dict_nom,
        calib_var="z",
        est_method="matrix",
        calib_method="alpha",
        print_output=False,
    )
    calib3.run()
    dfXst = pd.DataFrame(
        data=[
            [44.4005, 0.9519, 1.2050, 33.8055, 11.6913, 1.2971],
            [44.7632, 0.9526, 1.2014, 19.1578, 21.8479, 1.1553],
        ],
        columns=["R", "wR", "wS", "Q1", "Q2", "z"],
        index=["Q1_max", "Q2_max"],
    )
    dfphi = pd.DataFrame(
        data=[[0.8857, 0.9519], [0.8929, 0.9526]],
        columns=["R", "wR"],
        index=["Q1_max", "Q2_max"],
    )
    dfgamma = pd.DataFrame(
        data=[[1.2050, 0.9677, 0.9381], [1.2014, 0.9677, 0.9381]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    dfpsi = pd.DataFrame(
        data=[[1.0, 1.0, 0.5367], [1.0, 0.5651, 1.0]],
        columns=["wS", "Q1", "Q2"],
        index=["Q1_max", "Q2_max"],
    )
    vect_design_z3 = np.array([1.2980, 1.1537])
    vect_design_beta3 = np.array([3.7037, 4.2869])
    # validate results
    assert pytest.approx(calib3.dfXstarcal, abs=1e-4) == dfXst
    assert pytest.approx(calib3.df_phi, abs=1e-4) == dfphi
    assert pytest.approx(calib3.df_gamma, abs=1e-4) == dfgamma
    assert pytest.approx(calib3.df_psi, abs=1e-4) == dfpsi
    assert pytest.approx(calib3.get_design_param_factor(), abs=1e-4) == vect_design_z3
    assert (
        pytest.approx(calib3.calc_beta_design_param(np.max(vect_design_z3)), abs=1e-3)
        == vect_design_beta3
    )


def test_calibration_coeff_opt_3():
    """
    Perform SORM analysis
    """
    lc, dict_nom, betaT = setup3()
    calib1 = ra.Calibration(
        lc,
        target_beta=betaT,
        dict_nom_vals=dict_nom,
        calib_var="z",
        est_method="coeff",
        calib_method="optimize",
        print_output=False,
    )
    calib1.run()
    dfXst = pd.DataFrame(
        data=[
            [0.6194, 1.0194, 1.8722, 1.2591, 1.6108, 3.5045],
            [0.6137, 1.0202, 1.4497, 1.727 , 1.7667, 3.4546],
            [0.6124, 1.0207, 1.5489, 1.3686, 1.8671, 3.3951],
        ],
        columns=["R", "G", "Q1", "Q2", "Q3", "z"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    dfphi = pd.DataFrame(
        data=[[0.8005], [0.7931], [0.7915]], columns=["R"], 
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    dfgamma = pd.DataFrame(
        data=[[1.0194, 1.3634, 1.1072, 1.2269], 
            [1.0202, 1.3634, 1.1072, 1.2269], 
            [1.0207, 1.3634, 1.1072, 1.2269], 
              ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    dfpsi = pd.DataFrame(
        data=[
            [1.    , 1.    , 0.7291, 0.8627],
            [1.    , 0.7743, 1.    , 0.9463],
            [1.    , 0.8273, 0.7925, 1.    ],
            ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    calib1.print_detailed_output(precision=4)
    print(calib1.df_nom)
    print(calib1.calc_df_pgRS())
    print(calib1.get_design_param_factor())
    vect_design_z1 = np.array([3.5045, 3.4546, 3.3951])
    vect_design_beta1 = np.array([4.8, 4.8641, 4.9421])
    # validate results
    assert pytest.approx(calib1.dfXstarcal, abs=1e-4) == dfXst
    assert pytest.approx(calib1.df_phi, abs=1e-4) == dfphi
    assert pytest.approx(calib1.df_gamma, abs=1e-4) == dfgamma
    assert pytest.approx(calib1.df_psi, abs=1e-4) == dfpsi
    assert pytest.approx(calib1.get_design_param_factor(), abs=1e-4) == vect_design_z1
    assert (
        pytest.approx(calib1.calc_beta_design_param(np.max(vect_design_z1)), abs=1e-4)
        == vect_design_beta1
    )


def test_calibration_mat_opt_3():
    """
    Perform SORM analysis
    """
    lc, dict_nom, betaT = setup3()
    calib2 = ra.Calibration(
        lc,
        target_beta=betaT,
        dict_nom_vals=dict_nom,
        calib_var="z",
        est_method="matrix",
        calib_method="optimize",
        print_output=False,
    )
    calib2.run()
    dfXst = pd.DataFrame(
        data=[
            [0.6194, 1.0194, 1.8722, 1.2591, 1.6108, 3.5045],
            [0.6137, 1.0202, 1.4497, 1.7270 , 1.7667, 3.4546],
            [0.6124, 1.0207, 1.5489, 1.3686, 1.8671, 3.3951],
        ],
        columns=["R", "G", "Q1", "Q2", "Q3", "z"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    dfphi = pd.DataFrame(
        data=[[0.8005], [0.7931], [0.7915]], columns=["R"], 
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    dfgamma = pd.DataFrame(
        data=[[1.0194, 1.3634, 1.1072, 1.2269], 
            [1.0202, 1.3634, 1.1072, 1.2269], 
            [1.0207, 1.3634, 1.1072, 1.2269], 
              ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    dfpsi = pd.DataFrame(
        data=[
            [1.    , 1.    , 0.7778, 0.7997],
            [1.    , 0.8352, 1.    , 0.7997],
            [1.    , 0.8352, 0.7778, 1.    ],
            ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    vect_design_z2 = np.array([3.5045, 3.4546, 3.3951])
    vect_design_beta2 = np.array([4.8, 4.8641, 4.9421])
    # validate results
    assert pytest.approx(calib2.dfXstarcal, abs=1e-4) == dfXst
    assert pytest.approx(calib2.df_phi, abs=1e-4) == dfphi
    assert pytest.approx(calib2.df_gamma, abs=1e-4) == dfgamma
    assert pytest.approx(calib2.df_psi, abs=1e-4) == dfpsi
    assert pytest.approx(calib2.get_design_param_factor(), abs=1e-4) == vect_design_z2
    assert (
        pytest.approx(calib2.calc_beta_design_param(np.max(vect_design_z2)), abs=1e-4)
        == vect_design_beta2
    )


def test_calibration_mat_alpha_3():
    """
    Perform SORM analysis
    """
    lc, dict_nom, betaT = setup3()
    calib3 = ra.Calibration(
        lc,
        target_beta=betaT,
        dict_nom_vals=dict_nom,
        calib_var="z",
        est_method="matrix",
        calib_method="alpha",
        print_output=False,
    )
    calib3.run()
    dfXst = pd.DataFrame(
        data=[
            [0.6194, 1.0194, 1.8722, 1.2591, 1.6108, 3.5045],
            [0.6137, 1.0202, 1.4497, 1.7270 , 1.7667, 3.4546],
            [0.6124, 1.0207, 1.5489, 1.3686, 1.8671, 3.3951],
        ],
        columns=["R", "G", "Q1", "Q2", "Q3", "z"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    dfphi = pd.DataFrame(
        data=[[0.8005], [0.7931], [0.7915]], columns=["R"], 
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    dfgamma = pd.DataFrame(
        data=[[1.0194, 1.3634, 1.1072, 1.2269], 
            [1.0202, 1.3634, 1.1072, 1.2269], 
            [1.0207, 1.3634, 1.1072, 1.2269], 
              ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    dfpsi = pd.DataFrame(
        data=[
            [1.    , 1.    , 0.7778, 0.7997],
            [1.    , 0.8352, 1.    , 0.7997],
            [1.    , 0.8352, 0.7778, 1.    ],
            ],
        columns=["G", "Q1", "Q2", "Q3"],
        index=["Q1_max", "Q2_max", "Q3_max"],
    )
    vect_design_z3 = np.array([3.5045, 3.4546, 3.3951])
    vect_design_beta3 = np.array([4.8, 4.8641, 4.9421])
    # validate results
    assert pytest.approx(calib3.dfXstarcal, abs=1e-4) == dfXst
    assert pytest.approx(calib3.df_phi, abs=1e-4) == dfphi
    assert pytest.approx(calib3.df_gamma, abs=1e-4) == dfgamma
    assert pytest.approx(calib3.df_psi, abs=1e-4) == dfpsi
    assert pytest.approx(calib3.get_design_param_factor(), abs=1e-4) == vect_design_z3
    assert (
        pytest.approx(calib3.calc_beta_design_param(np.max(vect_design_z3)), abs=1e-4)
        == vect_design_beta3
    )
test_calibration_coeff_opt_3()

def get_psi_max(dfpsi):
    """
    Get :math:`\\psi` dataframe corresponding to maximum estimates of dfpsi.

    Parameters
    ----------
    dfpsi : DataFrame
        Dataframe of :math:`\\psi` per load case.

    Returns
    -------
    df_psi_max : DataFrame
        Dataframe of :math:`\\psi` corresponding to maximum of each load effect.

    """
    df_psi_max = dfpsi.copy()
    np.fill_diagonal(df_psi_max.values, 0.0)
    print(df_psi_max)
    df_psi_max = df_psi_max.clip(df_psi_max.max(), axis=1)
    return df_psi_max