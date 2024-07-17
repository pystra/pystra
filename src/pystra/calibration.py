# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from .distributions import Constant
from .model import LimitState, StochasticModel
from .form import Form
import matplotlib.pyplot as plt


class Calibration:
    r"""Class for calibrating partial and combination factors.

    The factors are: :math:`\\phi`, :math:`\\gamma`, and :math:`\\psi`
    factors for a given load combination instance and target reliability.

    Attributes
    ----------
    betaT : float
        Target reliability index
    calib_method : str
        The calibration algorithm to use
    cvar : str
        The label of the calibration variable.
    df_nom : DataFrame
        A dataframe of nominal values
    df_Xstar : DataFrame
        A dataframe of design point values
    df_phi : DataFrame
        A dataframe of partial factors for resistances
    df_gamma : DataFrame
        A dataframe of partial factors for loads
    df_psi : DataFrame
        A dataframe of load combination factors
    dict_nom : dict
        Dictionary of nominal values
    est_method : str
        The estimation method
    label_comb_vrs : str
        Labels of combination variables
    label_comb_cases : str
        Labels of combination load variables
    label_R : str
        Labels of resistance variables
    label_other : str
        Labels of other load variables
    label_all : str
        Labels of all variables including design parameter
    loadCombObj : LoadCombination
        LoadCombination object
    print_output : bool
        Whether or not to print output to the console
    """

    def __init__(
        self,
        loadcombobj,
        target_beta,
        dict_nom_vals,
        calib_var,
        calib_method="optimize",
        est_method="matrix",
        print_output=False,
    ):
        """
        Initialize class instance.

        Parameters
        ----------
        loadcombobj : Class object
            Class LoadCombination object.
        target_beta : Float
            Target reliability index for calibration.
        dict_nom_vals : Dictionary
            Dictionary of nominal values.
        calib_var : String
            Label of calibration variable in the LSF.
        calib_method : String, optional
            Calibration method for the analysis: "optimize" or "alpha".
            The default is "optimize".
        est_method : String, optional
            Estimation method for the factors: "matrix" or "coeff".
            The default is "matrix".
        print_output : Boolean, optional
            Boolean flag for printing analysis output. The default is False.

        Returns
        -------
        None.

        """
        # Instance attributes
        self.lc_obj = loadcombobj
        self.beta_t = target_beta
        self.calib_method = calib_method
        self.est_method = est_method
        self.print_output = print_output
        self.dict_nom = dict_nom_vals
        self.df_nom = pd.DataFrame(
            data=dict_nom_vals, index=loadcombobj.label_comb_cases
        )
        (
            self.label_R,
            self.label_other,
            self.label_comb_vrs,
            self.label_comb_cases,
            self.label_all,
        ) = self._set_labels()
        self.label_S = self.label_other + self.label_comb_vrs
        self.cvar = calib_var

    ## Utility Methods
    @staticmethod
    def _print_form_results(form):
        """
        Print the results for a Pystra FORM object

        Parameters
        ----------
        form : Pystra FORM object
            Pystra FORM object after running analysis.

        Returns
        -------
        None.

        """
        print(
            f" \n β = {form.getBeta():.3f} \n α = {form.getAlpha().round(3)}\
              \n x* = {form.getDesignPoint(False).round(3)}"
        )

    @staticmethod
    def _get_missing_element(mainlist, subsetlist):
        """
        Identify the element in the mainlist which is not present in the subset
        list.

        Parameters
        ----------
        mainlist : List

        subsetlist : List


        Returns
        -------
        missing : List
            List containing mainlist elements not present in subset list.

        """
        missing = [True if xx not in subsetlist else False for xx in mainlist]
        missing = list(np.array(mainlist)[missing])
        return missing

    def _set_labels(self):
        """
        Set labels of random variables and load cases based on the
        LoadCombination class Object.

        Returns
        -------
        label_R : List
            List of resistance variables.
        label_other : List
            List of other load variables.
        label_comb_vrs : List
            List of load combination case variables.
        label_comb_cases : List
            List of load combination cases.
        label_all : List
            List of all random variables.

        """
        label_comb_cases = self.lc_obj.get_label("comb_cases")
        label_R = self.lc_obj.get_label("resist")
        label_comb_vrs = self.lc_obj.get_label("comb_vrs")
        label_other = self.lc_obj.get_label("other")
        label_all = self.lc_obj.get_label("all")
        return label_R, label_other, label_comb_vrs, label_comb_cases, label_all

    def calc_lsf_eval_df(self, df, ret_abs=True):
        """Calculate the LSF evaluation of a Dataframe elements.

        Pass each dataframe value to the LSF and get corresponding LSF
        evaluation.

        The columns of the dataframe must be a subset of LSF arguments.

        Parameters
        ----------
        df : Dataframe
            Dataframe for element-wise LSF evaluation.
            len(df.index) can be greater-than or equal to 1
        ret_abs : Bool, optional
            Return abs of LSF evaluation. The default is True.

        Returns
        -------
        df_lsf : Dataframe
            Dataframe containing element-wise LSF evaluation of passed df.

        """

        df_lsf = df.copy()
        for col in df:
            jj = [
                self.lc_obj.eval_lsf_kwargs(**{"z": 1.0, col: xx}) for xx in df_lsf[col]
            ]
            df_lsf.loc[:, col] = [abs(xx) for xx in jj] if ret_abs else jj
        return df_lsf

    ## Projection Methods
    def _calibration_optimize(
        self,
        rel_func,
        z0,
        target_beta,
        print_output=False,
        xtol=1e-4,
        max_iter=None,
        **kwargs,
    ):
        """
        Calibrate design parameter for the supplied rel_func to target
        reliability index using optimization algorithm.

        Parameters
        ----------
        rel_func : Function
            Reliability function for the LoadCombination problem parameterized
            on the load case name and returning a FORM object.
        z0 : Float
            Initial value of the design parameter for resistance.
        target_beta : Float
            The target reliability index.
        print_output : Bool, optional
            Display output for print_outputging. The default is False.
        xtol : Float, optional
            Relative error tolerance for convergence. The default is 1e-4.
        max_iter : Integer, optional
            Maximum number of iterations for optimizations. The default is
            as per scipy.optimize.fsolve.
        **kwargs : Keyword arguments
            Keyword arguments for rel_func.

        Returns
        -------
        Zk_opt : Array
            Calibrated value of the design parameter for resistance at target_beta.
        form : Pystra FORM object
            The form object at beta_t

        """
        cvar = self.cvar

        def obj_func(Zk, beta_t):
            val = Constant(cvar, Zk)
            dict_z = {cvar: val}
            kwargs.update(dict_z)
            form = rel_func(**kwargs)
            if print_output:
                ## Change to inbuilt
                print(
                    f"\n{Zk=} \n β = {form.getBeta():.3f} \
                      \n α = {form.getAlpha()} \
                     \n x* = {form.getDesignPoint(False)}"
                )
            return beta_t - form.beta

        if max_iter is None:
            Zk_opt = fsolve(obj_func, x0=z0, args=(target_beta), xtol=xtol)
        else:
            Zk_opt = fsolve(
                obj_func, x0=z0, args=(target_beta), xtol=xtol, maxfev=max_iter
            )
        val = Constant(cvar, Zk_opt)
        dict_z = {cvar: val}
        kwargs.update(dict_z)
        form = rel_func(**kwargs)
        return Zk_opt, form

    def _calibration_alpha(
        self,
        rel_func,
        z0,
        target_beta,
        print_output=False,
        abstol=1e-4,
        max_iter=int(1e2),
        **kwargs,
    ):
        """
        Calibrate design parameter for the supplied rel_func to target
        reliability index using iterative :math:`\\alpha` projection method.

        Parameters
        ----------
        rel_func : Function
            Reliability function for the LoadCombination problem parameterized
            on the load case name and returning a FORM object.
        z0 : Float
            Initial value of the design parameter for resistance.
        target_beta : Float
            The target reliability index.
        print_output : Bool, optional
            Display output for print_outputging. The default is False.
        abstol : Float, optional
            Absolute error tolerance for convergence. The default is 1e-4.
        max_iter : Integer, optional
            Maximum number of iterations. The default 100.
        **kwargs : Keyword arguments
            Keyword arguments for rel_func.

        Returns
        -------
        Zk_opt : Float
            Calibrated value of the design parameter for resistance at target_beta.
        form : Pystra FORM object
            The form object at beta_t

        """
        ## Initialize algorithm
        cvar = self.cvar
        val = Constant(cvar, z0)
        dict_z = {cvar: val}
        kwargs.update(dict_z)
        form0 = rel_func(**kwargs)
        alpha0 = form0.getAlpha()
        n_iter = 0
        beta0 = form0.getBeta()
        alpha_cal = alpha0
        form_cal = form0
        beta_cal = beta0
        z_cal = z0
        columns = self._get_df_Xstar_labels(form0)
        if print_output:
            print(f"\n ==== Iteration {n_iter} ====")
            self._print_form_results(form0)
        ## Iterate till convergence
        while not np.isclose(beta_cal, self.beta_t, atol=abstol):
            ## U-space projection
            U_cal = alpha_cal * self.beta_t
            ## X-space projection
            Xstar_cal = form_cal.transform.u_to_x(
                U_cal, form_cal.model.getMarginalDistributions()
            )
            ## Calculate the design parameter for the Calibrated LSF
            dfXst_cal = pd.DataFrame(data=[Xstar_cal], columns=columns)
            z_cal = np.array([self.calc_design_param_Xst(dfXst_cal)])
            ## Check Calibrated reliability index
            val = Constant(cvar, z_cal)
            dict_z = {cvar: val}
            kwargs.update(dict_z)
            form_cal = rel_func(**kwargs)
            beta_cal = form_cal.getBeta()
            alpha_cal = form_cal.getAlpha()
            ## New U-space projection
            U_cal = alpha_cal * self.beta_t
            n_iter += 1  ## Increment number of iterations
            if print_output:
                print(f"\n ==== Iteration {n_iter} ====")
                self._print_form_results(form_cal)
                print(f"\n z = {z_cal}")
            if n_iter == max_iter:
                print(f"Maximum iterations {max_iter} exceeded! Aborting!")
                break
        return z_cal, form_cal

    def calc_design_param_Xst(self, dfXst):
        """
        Calculate design parameter for resistance from design points.

        Parameters
        ----------
        dfXst : Dataframe
            Dataframe containing design points.
            Note: len(dfXst.index) = 1

        Returns
        -------
        z : Float
            design parameter for resistance corresponding to the design pt.

        """
        dfS = dfXst[self.label_other + self.label_comb_vrs]
        dfS_dict = dfS.to_dict("records")[0]
        sum_loadeff = self.lc_obj.eval_lsf_kwargs(**dfS_dict)
        R_dict = dfXst[self.label_R].to_dict("records")[0]
        sum_resist = self.lc_obj.eval_lsf_kwargs(**R_dict)
        z = sum_loadeff / sum_resist
        z = float(abs(z))
        return z

    def _get_df_Xstar(self, list_form_obj, cols=None, idx=None):
        """
        Get a dataframe of design points in physical space using a list
        of FORM objects

        Parameters
        ----------
        list_form_obj : List
            List of FORM objects.
        cols : List or pandas.DataFrame.columns
            Column values for output Dataframe. Default is
            list_form_obj[0].model.getNames()[1:]
        idx : List or pandas.DataFrame.index
            Index values for output Dataframe. Default is integer array.

        Returns
        -------
        dfXstar : Dataframe
            Dataframe of design points in physical space.

        """
        Xstar = [xx.getDesignPoint(uspace=False) for xx in list_form_obj]
        label_vrs = self._get_df_Xstar_labels(list_form_obj[0])
        cols = label_vrs if cols is None else cols
        idx = np.arange(len(list_form_obj)) if idx is None else idx
        dfXstar = pd.DataFrame(data=Xstar, columns=cols, index=idx)
        return dfXstar

    def _get_df_Xstar_labels(self, form):
        """
        Get labels for the DataFrame of design points using the form objects.

        The Labels contains the [resistance variables + other load variables
        + load combination variables]

        Parameters
        ----------
        form : Object
            Pystra FORM object.

        Returns
        -------
        label_vrs : list
            Labels for the DataFrame of design points using the form objects.

        """
        label_const = form.model.getConstants().keys()
        label_all = form.model.getNames()
        label_vrs = sorted(list(set(label_all) - set(label_const)), key=label_all.index)
        return label_vrs

    def run(self, est_method=None, set_max=False):
        """
        Run calibration analysis to estimate :math:`\\phi`, :math:`\\gamma`,
        and :math:`\\psi` factors, and set class dataframe attributes
        corresponding to each factor.

        Parameters
        ----------
        est_method : String, optional
            Calibration method override. The default is "matrix".

        Returns
        -------
        None.

        """
        est_method = self.est_method if est_method is None else est_method
        self._run_calibration()
        if est_method == "coeff":
            self.df_phi, self.df_gamma, self.df_psi = self._estimate_factors_coeff(
                set_max
            )
        elif est_method == "matrix":
            self.df_phi, self.df_gamma, self.df_psi = self._estimate_factors_matrix(
                set_max
            )

    def _run_calibration(self):
        """
        Run the method for calibrating the design parameter to the target
        reliability index.

        Returns
        -------
        None.

        """
        arr_zcal, list_form_cal = self._calibrate_design_param()
        self.dfXstarcal = self._get_df_Xstar(
            list_form_cal, idx=self.lc_obj.label_comb_cases
        )
        self.dfXstarcal["z"] = arr_zcal

    def _estimate_factors_coeff(self, set_max=False):
        """
        Estimate the factors :math:`\\phi`, :math:`\\gamma`, and
        :math:`\\psi` factors using the coefficient approach.

        Parameters
        ----------
        set_max : Boolean, optional
            Set psi estimates to their corresponding max values. The default
            is False.

        Returns
        -------
        df_phi : Dataframe
            Dataframe of :math:`\\phi` per load case.
        df_gamma : Dataframe
            Dataframe of :math:`\\gamma`.
        df_psi : Dataframe
            Dataframe of :math:`\\psi` per load case.

        """
        df_phi, df_gamma, df_psi = self.calc_pg_coeff(
            self.dfXstarcal, print_output=self.print_output
        )
        df_psi = self.get_psi_max(df_psi) if set_max else df_psi
        df_phi = self.get_phi_min(df_phi) if set_max else df_phi
        return df_phi, df_gamma, df_psi

    def get_psi_max(self, dfpsi):
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
        df_psi_max = dfpsi[self.label_comb_vrs].copy()
        np.fill_diagonal(df_psi_max.values, 0.0)
        df_psi_max = df_psi_max.clip(df_psi_max.max(), axis=1)
        np.fill_diagonal(df_psi_max.values, 1.0)
        if len(self.label_other) > 0:
            df_psi_max.loc[:, self.label_other] = dfpsi[self.label_other]
        return df_psi_max

    def _calibrate_design_param(self):
        """
        Calibrate design parameter for resistance to target Beta for all
        load combination cases using the specified projection method. The
        starting value of the calibration variable is taken as that specified
        in the LoadCombination object definition.

        Returns
        -------
        list_z_cal : List
            List of calibrated design parameters per load comb case.
        list_form_cal : List
            List of calibrated Pystra FORM objects per load comb case.

        """
        startz = self.lc_obj.constant[self.cvar].getValue()
        rel_func = self.lc_obj.run_reliability_case
        list_z_cal = []
        list_form_cal = []
        for lc in self.lc_obj.label_comb_cases:
            if self.calib_method == "optimize":
                zcal, form = self._calibration_optimize(
                    rel_func,
                    z0=startz,
                    print_output=self.print_output,
                    target_beta=self.beta_t,
                    lcn=lc,
                )
            elif self.calib_method == "alpha":
                zcal, form = self._calibration_alpha(
                    rel_func,
                    z0=startz,
                    print_output=self.print_output,
                    target_beta=self.beta_t,
                    lcn=lc,
                )
            list_z_cal.append(zcal)
            list_form_cal.append(form)
        list_z_cal = np.concatenate(list_z_cal)
        arr_beta = np.array([xx.getBeta() for xx in list_form_cal])
        if self.print_output:
            print(f"\n Calibrated reliabilities = {arr_beta}")
        return list_z_cal, list_form_cal

    def _estimate_factors_matrix(self, set_max=False):
        """
        Estimate the factors :math:`\\phi`, :math:`\\gamma`, and :math:`\\psi` factors using the
        Matrix approach.

        Returns
        -------
        df_phi : Dataframe
            Dataframe of :math:`\\phi` per load case.
        df_gamma : Dataframe
            Dataframe of :math:`\\gamma`.
        df_psi : Dataframe
            Dataframe of :math:`\\psi`.

        """
        df_phi, df_gamma, df_psi = self.calc_pg_matrix(
            self.dfXstarcal, print_output=self.print_output
        )
        df_phi = self.get_phi_min(df_phi) if set_max else df_phi
        df_psi = self.get_psi_max(df_psi) if set_max else df_psi
        return df_phi, df_gamma, df_psi

    def calc_pg_coeff(self, dfXst, print_output=False):
        """
        Calculate :math:`\\phi`, :math:`\\gamma`, and :math:`\\psi` for the given set of design
        points and nominals using comparison of design pt coefficients approach.

        Parameters
        ----------
        dfXst : Dataframe
            Dataframe containing all design points at target reliability.
        print_output : Boolean, optional
            print_output flag for displaying intermediate and final output of function.
            The default is False.

        Returns
        -------
        df_phi : Dataframe
            Dataframe containing :math:`\\phi` estimates for resistance variables
            per load case.
        df_gamma : Dataframe
            Dataframe containing :math:`\\gamma` estimates for all static and
            combination load variables per load case.
        df_psi : Dataframe
            Dataframe containing :math:`\\psi` estimates for all static and
            combination load variables per load case.

        """
        ## Estimate :math:`\\phi` and :math:`\\gamma`
        df_Xst_nom = self.calc_Xst_nom(dfXstar=dfXst)
        df_phi = self.calc_phi(df_Xst_nom)
        df_gamma_static, df_gamma_comb = self.calc_gamma(df_Xst_nom)
        df_gamma = pd.concat((df_gamma_static, df_gamma_comb), axis=1)
        ## Estimate :math:`\\psi`
        df_psi = dfXst / df_gamma / self.df_nom
        df_psi = df_psi[self.label_S]
        if print_output:
            print(f"\n $\\phi$, \n {df_phi}")
            print(f"\n $\\gamma$ static, \n {df_gamma_static}")
            print(f"\n $\\gamma$ comb vrs, \n {df_gamma_comb}")
            print(f"\n psi, \n {df_psi}")
        return df_phi, df_gamma, df_psi

    def calc_Xst_nom(self, dfXstar):
        """
        Calculate the design point DataFrame divided by the nominal values
        per load case and adjust for :math:`\\psi` factors for combination
        load variables.

        Parameters
        ----------
        dfXstar : DataFrame
            DataFrame containing all design points at target reliability for
            all load cases.

        Returns
        -------
        df_Xst_nom : DataFrame
            Design point DataFrame factored by the nominal values
            per load case.

        """
        df_Xst_nom = dfXstar / self.df_nom

        # Adjust for :math:`\\psi` factors; replace with non :math:`\\psi`
        # entries
        for comb, vrs in self.lc_obj.dict_comb_cases.items():
            other_combs = set(self.label_comb_cases) - set([comb])
            gamma = df_Xst_nom.loc[comb, vrs]
            df_Xst_nom.loc[list(other_combs), vrs] = gamma.values
        df_Xst_nom = df_Xst_nom[dfXstar.columns]
        return df_Xst_nom

    def calc_phi(self, dfXstnom):
        """
        Calculate resistance factors :math:`\\phi` from a dataframe of design points
        factored by the nominal values.

        Parameters
        ----------
        dfXstnom : DataFrame
            Design point DataFrame factored by the nominal values
            per load case.
        set_max : Boolean, optional
            Set :math:`\\phi` estimates to their corresponding max values. The default
            is False.

        Returns
        -------
        df_phi : DataFrame
            Resistance factors :math:`\\phi` for resistance variables per load case.

        """
        df_phi = dfXstnom[self.label_R]
        return df_phi

    def get_phi_min(self, dfphi_):
        dfphi = dfphi_.copy()
        dfphi = dfphi.clip(upper=dfphi.min(), axis=1)
        return dfphi

    def calc_gamma(self, dfXstnom):
        """
        Calculate Load factors :math:`\\gamma` from a dataframe of design points
        factored by the nominal values.

        Parameters
        ----------
        dfXstnom : DataFrame
            Design point DataFrame factored by the nominal values
            per load case.

        Returns
        -------
        dfgamma_static : DataFrame
            Load factors for static variables per load case.
        dfgamma_comb : DataFrame
            Load factors for combination variables per load case.

        """
        dfgamma_static = dfXstnom[self.label_other]
        dfgamma_comb = dfXstnom[self.label_comb_vrs]
        return dfgamma_static, dfgamma_comb

    def calc_pg_matrix(self, dfXst, print_output=False):
        """
        Calculate :math:`\\phi`, :math:`\\gamma`, and :math:`\\psi` for
        the given set of design points and nominals using the Matrix approach.

        Parameters
        ----------
        dfXst : Dataframe
            Dataframe containing all design points at target reliability.
        print_output : Boolean, optional
            print_output flag for displaying intermediate and final output of function.
            The default is False.

        Returns
        -------
        df_phi : Dataframe
            Dataframe containing :math:`\\phi` estimates for resistance variables
            per load case.
        df_gamma : Dataframe
            Dataframe containing :math:`\\gamma` estimates for all static and
            combination load variables per load case.
        df_psi : Dataframe
            Dataframe containing :math:`\\psi` estimates for all static and
            combination load variables per load case.

        """
        ## Estimate :math:`\\phi` and :math:`\\gamma`
        df_Xst_nom = self.calc_Xst_nom(dfXstar=dfXst)
        df_phi = self.calc_phi(df_Xst_nom)
        df_gamma_static, df_gamma_comb = self.calc_gamma(df_Xst_nom)

        df_gamma = pd.concat((df_gamma_static, df_gamma_comb), axis=1)
        ## Estimate :math:`\\psi`
        # Get RHS :math:`\\phi~R~z-\\gamma_g~G-\\gamma_i~S_i`
        phiRz_egS = self.calc_phiRz_egS_vect(dfXst)
        # Get LHS :math:`\\gamma_j~S_j`
        df_gamma_nom = pd.concat([df_phi, df_gamma], axis=1) * self.df_nom
        epgS_mat = self.calc_epgS_mat(df_gamma_nom)
        # Estimate
        psi = np.linalg.solve(epgS_mat, phiRz_egS)
        psi_mat = self._get_psi_row_mat(len(self.label_other), psi)
        df_psi = pd.DataFrame(
            data=psi_mat, columns=df_gamma.columns, index=df_gamma.index
        )
        if self.print_output:
            print(f"\n $\\phi$, \n {df_phi}")
            print(f"\n $\\gamma$ static, \n {df_gamma_static}")
            print(f"\n $\\gamma$ comb vrs, \n {df_gamma_comb}")
            print(f"\n egS Matrix, \n {epgS_mat}")
            print(f"\n zpR-gS Vector, \n {phiRz_egS}")
            print(f"\n psi, \n {df_psi}")
        return df_phi, df_gamma, df_psi

    def calc_phiRz_egS_vect(self, dfXstar):
        """
        Get RHS for matrix estimation method,
        :math:`\\phi~R~z-\\gamma_g~G-\\gamma_i~S_i`

        Parameters
        ----------
        dfXstar : Dataframe
            Dataframe containing all design points at target reliability.

        Returns
        -------
        phiRz_egS_vect : Array
            RHS for matrix estimation method as 1D Array.

        """
        ## Initialize the vector
        phiRz_egS_vect = np.zeros(len(dfXstar.index))
        idx = 0
        for comb in dfXstar.index:
            # Get RVs with cvar except the other combination variable(s)
            s_label = self.lc_obj.dict_comb_cases[comb]
            s_other = set(self.label_comb_vrs) - set(s_label)
            label_all_rvs = (
                self.label_R + self.label_comb_vrs + self.label_other + [self.cvar]
            )
            list_others = list(set(label_all_rvs) - s_other)
            # Pass RVs except the other combination variable(s) to the LSF
            dfXstar_dict = dfXstar.loc[[comb], list_others].to_dict("records")[0]
            phiRz_egS_vect[idx] = self.lc_obj.eval_lsf_kwargs(**dfXstar_dict)
            idx += 1
        return phiRz_egS_vect

    def calc_epgS_mat(self, dfgammanom):
        """Get LHS for matrix estimation method, :math:`\\gamma_j~S_j`.
        The LHS is evaluated by evaluating the LSF with appropriate random
        variables to account for any constant multipliers. The implementation
        works for both, linear and non-linear LSFs. For more algorithmic
        details, ref to Appendix A, Caprani and Khan, Structural Safety, 2023.

        Parameters
        ----------
        dfgammanom : Dataframe
            Dataframe containing product of nominal values and safety factors,
            along with calibrated z values.

        Returns
        -------
        epgS_mat : Array
            LHS for matrix estimation method as 2D Array.

        """

        ## Initialize the vector
        epgS_mat = np.zeros((len(dfgammanom.index), len(self.label_comb_vrs)))
        idx = 0
        for comb in dfgammanom.index:
            # Get load comb RV with other RVs
            s_label = self.lc_obj.dict_comb_cases[comb]
            rvs_for_lhs = list(set(self.label_other) | set(s_label))
            # Pass load comb RV with other RVs to the LSF
            dfXstar_dict_comb = dfgammanom.loc[[comb], rvs_for_lhs].to_dict("records")[
                0
            ]
            if len(self.label_other) > 0:
                dfXstar_dict_other = dfgammanom.loc[[comb], self.label_other].to_dict(
                    "records"
                )[0]
            else:
                dfXstar_dict_other = {}
            epgS_mat[:, idx] = self.lc_obj.eval_lsf_kwargs(
                **dfXstar_dict_comb
            ) - self.lc_obj.eval_lsf_kwargs(**dfXstar_dict_other)
            idx += 1
        epgS_mat = epgS_mat * -1
        np.fill_diagonal(epgS_mat, 0)
        return epgS_mat

    def _get_psi_row_mat(self, num_other_vrs, psi_comb_vrs):
        """
        Convert :math:`\\psi` estimates for load case variables into :math:`\\psi` matrix for all
        random variables (including non load case, i.e. other, variables). Each
        row of the output matrix corresponds to one set of :math:`\\psi` for all rvs
        for a load case. The value of :math:`\\psi` for non load case rvs is set to be 1.0.

        Parameters
        ----------
        num_other_vrs : Integer
            Number of other random variables (i.e. load effects not part of load
                                              combinations).
        psi_comb_vrs : Array or Ndarray
            If :math:`\\psi` is specified for all load combinations, then Array.
            If :math:`\\psi` is specified per load combination, then Ndrray Matrix.

        Returns
        -------
        psi_row_mat_ : Ndarray
            :math:`\\psi` Matrix for all rvs and combinations. Each row of Matrix
            corresponds to one set of :math:`\\psi` for all rvs for a load case.

        """
        num_comb_vrs = len(psi_comb_vrs)
        if len(psi_comb_vrs.shape) == 1:
            psi_row_mat_comb = psi_comb_vrs * np.ones((num_comb_vrs, num_comb_vrs))
            np.fill_diagonal(psi_row_mat_comb, 1)
        else:
            psi_row_mat_comb = psi_comb_vrs
        psi_row_mat_other = np.ones((num_comb_vrs, num_other_vrs))
        psi_row_mat_ = np.concatenate([psi_row_mat_other, psi_row_mat_comb], axis=1)
        return psi_row_mat_

    # Projection Method

    def calc_beta_design_param(self, design_z):
        """
        Calculate reliabilities based on given design parameter for resistance
        for all load combination cases.

        Parameters
        ----------
        design_z : Float
            design parameter for resistance

        Returns
        -------
        arr_beta : Array
            Array containing reliability indices corresponding to design_z for
            each load combination case.

        """
        cvar = self.cvar
        val = Constant(cvar, design_z)
        dict_z = {cvar: val}
        list_form_des = [
            self.lc_obj.run_reliability_case(lcn=xx, **dict_z)
            for xx in self.lc_obj.label_comb_cases
        ]
        arr_beta = np.array([xx.getBeta() for xx in list_form_des])
        if self.print_output:
            print(f"\n Design reliabilities = {arr_beta}")
        return arr_beta

    def calc_df_pgRS(self, min_phi, max_psi):
        """
        Calculate the DataFrame of all resistance and load variables nominal
        values multiplied by their respective factors, :math:`\\phi`, :math:`\\gamma`,
        and :math:`\\psi`, for all load cases.

        Returns
        -------
        df_pgRS : DataFrame


        """
        df_pgRS = self.df_nom.copy()
        df_phi = self.get_phi_min(self.df_phi) if min_phi else self.df_phi
        df_psi = self.get_psi_max(self.df_psi) if max_psi else self.df_psi
        df_gamma = self.df_gamma.max()
        df_pgRS.loc[:, self.label_S] = df_pgRS[self.label_S] * df_gamma * df_psi
        df_pgRS.loc[:, self.label_R] = df_pgRS[self.label_R] * df_phi
        return df_pgRS

    def get_design_param_factor(self, min_phi=True, max_psi=True):
        """
        Estimate the resistance design parameter for a given set of safety and
        combination factors, and nominals.

        Returns
        -------
        array_z : Array
            Array containing design parameters for all load combination cases.

        """
        df_pgRS = self.calc_df_pgRS(min_phi, max_psi)
        list_cols = [df_pgRS.loc[[xx], :] for xx in self.label_comb_cases]
        array_z = np.array([self.calc_design_param_Xst(xx) for xx in list_cols])
        return array_z

    def print_detailed_output(self, precision=2):
        """
        Print detailed outputs for Pystra Calibration, including calculations
        from intermediate steps.

        Parameters
        ----------
        precision : float, optional
            Decimal precision for roundingo off output. The default is 2.

        Returns
        -------
        None.

        """
        n = 54
        print("\n")
        print("=" * n)
        print("X* = \n", self.dfXstarcal.round(precision))
        print("\nphi = ", "\n", self.df_phi.round(precision))
        print("\ngamma =", "\n", self.df_gamma.round(precision))
        print("\npsi = ", "\n", self.df_psi.round(precision))
        print("=" * n)


class GenericModel:
    """
    A probability model for generic calibration.
    """

    def __init__(self):
        """
        Initialize the class which holds all the probability model info.

        Returns
        -------
        None.

        """
        self.phi = None
        self.gamma_g = None
        self.gamma_q = None
        self.gamma_p = None
        self.wR = None
        self.wS = None
        self.R = None
        self.G = None
        self.P = None
        self.Q = None
        self.Rk = None
        self.Gk = None
        self.Pk = None
        self.Qk = None

    def set_factors(self, phi, gamma_g, gamma_p, gamma_q):
        """
        Assign the partial factors to the generic probability model.

        Parameters
        ----------
        phi : float
            Capacity reduction factor.
        gamma_g : float
            Dead load factor.
        gamma_p : float
            Permanent load factor.
        gamma_q : float
            Live load factor.

        Returns
        -------
        None.

        """
        self.phi = phi
        self.gamma_g = gamma_g
        self.gamma_p = gamma_p
        self.gamma_q = gamma_q

    def set_distributions(self, wR, wS, R, G, P, Q):
        """
        Assign the PySTRA distribution objects

        Parameters
        ----------
        wR : PySTRA distribution object
            Model error for resistance.
        wS : PySTRA distribution object
            Model error for loading/analysis.
        R : PySTRA distribution object
            Resistance distribution.
        G : PySTRA distribution object
            Dead load distribution.
        P : PySTRA distribution object
            Permanent load distribution.
        Q : PySTRA distribution object
            Live load distribution.

        Returns
        -------
        None.

        """
        self.wR = wR
        self.wS = wS
        self.R = R
        self.G = G
        self.P = P
        self.Q = Q

    def set_nominals(self, Rk, Gk, Pk, Qk):
        """
        Set the nominal values of the parameters for use in design.

        Parameters
        ----------
        Rk : float
            Characteristic value of resistance.
        Gk : float
            Characteristic value of dead load.
        Pk : float
            Characteristic value of permanent load.
        Qk : float
            Characteristic value of live load.

        Returns
        -------
        None.

        """
        self.Rk = Rk
        self.Gk = Gk
        self.Pk = Pk
        self.Qk = Qk

    def set_model_errors(self, wR, wS):
        """
        Set the model errors

        Parameters
        ----------
        wR : PySTRA distribution object
            Model error for resistance.
        wS : PySTRA distribution object
            Model error for loading/analysis.

        Returns
        -------
        None.

        """

        self.wR = wR
        self.wS = wS

    def set_resistance_params(self, R, Rk, phi):
        """
        Set the resistance model parameters.

        Parameters
        ----------
        R : PySTRA distribution object
            Resistance distribution.
        Rk : float
            Characteristic value of resistance.
        phi : float
            Capacity reduction factor.

        Returns
        -------
        None.

        """

        self.R = R
        self.Rk = Rk
        self.phi = phi

    def set_dead_params(self, G, Gk, gamma_g):
        """
        Set the dead load model parameters.

        Parameters
        ----------
        G : PySTRA distribution object
            Dead load distribution.
        Gk : float
            Characteristic value of dead load.
        gamma_g : float
            Partial factor for dead load.

        Returns
        -------
        None.

        """

        self.G = G
        self.Gk = Gk
        self.gamma_g = gamma_g

    def set_permanent_params(self, P, Pk, gamma_p):
        """
        Set the permanent load model parameters.

        Parameters
        ----------
        P : PySTRA distribution object
            Permanent load distribution.
        Pk : float
            Characteristic value of permanent load.
        gamma_p : float
            Partial factor for permanent load.

        Returns
        -------
        None.

        """

        self.P = P
        self.Pk = Pk
        self.gamma_p = gamma_p

    def set_live_params(self, Q, Qk, gamma_q):
        """
        Set the dead load model parameters.

        Parameters
        ----------
        Q : PySTRA distribution object
            Live load distribution.
        Qk : float
            Characteristic value of live load.
        gamma_q : float
            Partial factor for live load.

        Returns
        -------
        None.

        """

        self.Q = Q
        self.Qk = Qk
        self.gamma_q = gamma_q

    def get(self):
        """
        Return a tuple of all the probability model parameters

        Returns
        -------
        phi : float
            Capacity reduction factor.
        gamma_g : float
            Dead load factor.
        gamma_p : float
            Permanent load factor.
        gamma_q : float
            Live load factor.
        wR : PySTRA distribution object
            Model error for resistance.
        wS : PySTRA distribution object
            Model error for loading/analysis.
        R : PySTRA distribution object
            Resistance distribution.
        G : PySTRA distribution object
            Dead load distribution.
        P : PySTRA distribution object
            Permanent load distribution.
        Q : PySTRA distribution object
            Live load distribution.
        Rk : float
            Characteristic value of resistance.
        Gk : float
            Characteristic value of dead load.
        Pk : float
            Characteristic value of permanent load.
        Qk : float
            Characteristic value of live load.

        """
        return (
            self.phi,
            self.gamma_g,
            self.gamma_p,
            self.gamma_q,
            self.wR,
            self.wS,
            self.R,
            self.G,
            self.P,
            self.Q,
            self.Rk,
            self.Gk,
            self.Pk,
            self.Qk,
        )


class GenericCalibration:
    """
    A generic code calibration
    """

    def __init__(self, aq_points, ag_points):
        """
        Initialize the generic code calibration class with the calculation points.

        Parameters
        ----------
        q_points : nd.array
            The grid of points to use for the live load ratio.
        g_points : nd.array
            The grid of points to use for the dead load ratio.

        Returns
        -------
        None.

        """
        self.aq_points = aq_points
        self.ag_points = ag_points
        self.prob_models = []
        self.is_analysed = False

    def add_model(self, model, color, label):
        """
        Add a probability model to the calibration.

        Parameters
        ----------
        model : GenericModel object
            A probability model.
        color : str
            Matplotlib string representation of a colour for plotting.
        label : str
            Label for use in plot legend.

        Returns
        -------
        None.

        """
        self.is_analysed = False
        model_dict = {"model": model, "betas": None, "color": color, "label": label}
        self.prob_models.append(model_dict)

    def lsf(self, z, aq, ag, wR, R, wS, G, P, Q):
        """
        Limit state function for the generic code calibration problem.

        Parameters
        ----------
        z : float
            Design parameter.
        aq : float
            Ratio of live to total loads.
        ag : float
            Ratio of dead to total dead and permanent load.
        wR : float
            Model error for resistance.
        R : float
            Normalized resistance.
        wS : float
            Model error for loading/analysis.
        G : float
            Normalized dead load.
        P : float
            Normalized permanent load.
        Q : float
            Normalized live load.

        Returns
        -------
        float
            Evaluation of the limit state.

        """
        return z * wR * R - wS * ((1 - aq) * (ag * G + (1 - ag) * P) + aq * Q)

    def design(self, aq, ag, phi, gamma_g, gamma_p, gamma_q, Rk, Gk, Pk, Qk):
        """
        Do a structural design for z, the design parameter, based on the code rules.

        Parameters
        ----------
        aq : float
            Ratio of live to total loads.
        ag : float
            Ratio of dead to total dead and permanent load.
        phi : float
            Capacity reduction factor.
        gamma_g : float
            Dead load factor.
        gamma_p : float
            Permanent load factor.
        gamma_q : float
            Live load factor.
        Rk : float
            Characteristic value of resistance.
        Gk : float
            Characteristic value of dead load.
        Pk : float
            Characteristic value of permanent load.
        Qk : float
            Characteristic value of live load.

        Returns
        -------
        z : float
            Design parameter.

        """
        z = (1 / (phi * Rk)) * (
            (1 - aq) * (ag * gamma_g * Gk + (1 - ag) * gamma_p * Pk) + aq * gamma_q * Qk
        )
        return z

    def analyse_design(self, z, aq, ag, wR, R, wS, G, P, Q):
        """
        Reliability analysis of the design according to the code rules.

        Parameters
        ----------
        z : float
            Design parameter.
        aq : float
            Ratio of live to total loads.
        ag : float
            Ratio of dead to total dead and permanent load.
        wR : float
            Model error for resistance.
        R : float
            Normalized resistance.
        wS : float
            Model error for loading/analysis.
        G : float
            Normalized dead load.
        P : float
            Normalized permanent load.
        Q : float
            Normalized live load.

        Returns
        -------
        float
            The reliability index, beta.

        """
        limit_state = LimitState(self.lsf)

        stochastic_model = StochasticModel()
        stochastic_model.addVariable(wR)
        stochastic_model.addVariable(R)
        stochastic_model.addVariable(wS)
        stochastic_model.addVariable(G)
        stochastic_model.addVariable(P)
        stochastic_model.addVariable(Q)

        stochastic_model.addVariable(Constant("z", z))
        stochastic_model.addVariable(Constant("aq", aq))
        stochastic_model.addVariable(Constant("ag", ag))

        form = Form(
            stochastic_model=stochastic_model,
            limit_state=limit_state,
        )
        form.run()
        return form.getBeta()

    def get_reliabilities(self, Aq, Ag, model):
        """
        Determine the reliabilities across the full range of dead and live load ratios.

        Parameters
        ----------
        Aq : nd.array
            The grid of live load ratio points.
        Ag : nd.array
            The grid of dead load ratio points.
        model : GenericModel
            Probability model and code rules for analysis.

        Returns
        -------
        beta : nd.array
            Reliability indices at each dead and live load ratio grid point.

        """
        beta = np.zeros((Ag.size, Aq.size))
        for i, ag in enumerate(Ag):
            for j, aq in enumerate(Aq):
                (
                    phi,
                    gamma_g,
                    gamma_p,
                    gamma_q,
                    wR,
                    wS,
                    R,
                    G,
                    P,
                    Q,
                    Rk,
                    Gk,
                    Pk,
                    Qk,
                ) = model.get()
                z = self.design(aq, ag, phi, gamma_g, gamma_p, gamma_q, Rk, Gk, Pk, Qk)
                beta[i, j] = self.analyse_design(z, aq, ag, wR, R, wS, G, P, Q)
        return beta

    def analyse(self):
        """
        Analyse the code calibration problem given the probability models.

        Raises
        ------
        ValueError
            If there are no probability models assigned.

        Returns
        -------
        None.

        """
        if not self.prob_models:
            raise ValueError("No probability models assigned")

        for pm in self.prob_models:
            pm["betas"] = self.get_reliabilities(
                self.aq_points, self.ag_points, pm["model"]
            )

        self.is_analysed = True

    def add_range(self, ax, xl, xu, ytext, text, alpha):
        def range_text(ax, x1, y1, x2, y2, text):
            ax.annotate(
                text="",
                xy=(x1, y1),
                xytext=(x2, y2),
                arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
            )
            ax.text(
                x1 + (x2 - x1) / 2,
                1.01 * y1 + (y2 - y1) / 2,
                text,
                ha="center",
                va="bottom",
            )

        ax.axvspan(xl, xu, facecolor="k", alpha=alpha)
        range_text(ax, xl, ytext, xu, ytext, text)

    def plot_region(self, ax, x, beta, facecolor, label):
        """
        Plots a region of reliability indices, and adds a line for the smallest
        dead load ratio.

        Parameters
        ----------
        ax : Matplotlib Axes object
            The axes to plot on.
        x : nd.array
            Vector of live load ratio points.
        beta : nd.array
            Matrix of reliability indices correspond go the dead and live ratios grid.
        facecolor : str
            Matplotlib string description of a colour.
        label : str
            Label for the plot legend.

        Returns
        -------
        None.

        """
        ax.fill_between(
            x,
            beta.min(axis=0),
            beta.max(axis=0),
            alpha=0.5,
            facecolor=facecolor,
            label=label,
        )
        ax.plot(x, beta[-1], color=facecolor, ls=":")

    def beta_region_plot(self, Aq, data, beta_t, ranges, figsize=(12, 9), **kwargs):
        """
        The main plotting function.

        Parameters
        ----------
        Aq : nd.array
            Vector of live load ratio points..
        data : List[Dict{GemericModel}]
            List of dictionaries of GenericModels, plotting info, and results.
        beta_t : float
            Target reliability index.
        ranges : Dict
            A dictionary of range information with the following keys:
                xl, xu, ytext, text, alpha
            defining the lower and upper aq values, the position of the text, the text,
            and the alpha of the range colouring.
        figsize : Tuple(float), optional
            The figure size in inches. The default is (12, 9).

        Returns
        -------
        fig : Matplotlib Figure object
            The figure.
        ax : Matplotlib Axes object
            The axes.

        """
        x = Aq

        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        ax.set_axisbelow(True)
        ax.grid(which="both", ls=":", color="k")

        if ranges:
            for r in ranges:
                self.add_range(ax, **r)

        for d in data:
            self.plot_region(ax, x, d["betas"], d["color"], d["label"])
        ax.axhline(beta_t, color="k", ls="--")
        t = ax.text(0.01, 0.99 * beta_t, rf"$\beta_T = {beta_t}$", ha="left", va="top")
        t.set_bbox(dict(edgecolor="w", facecolor="w", alpha=0.8, pad=0))
        ax.set_xlabel(r"Variable load ratio, $a_q = Q/(G+P+Q)$")
        ax.set_ylabel(r"Reliability index, $\beta$")
        ax.legend(loc="upper left")
        return fig, ax

    def plot(self, beta_t=4.7, ranges=None, **kwargs):
        """
        Plot the region for each probability model with their descriptions.

        Parameters
        ----------
        beta_t : float
            Target reliability index.
        ranges : Dict
            A dictionary of range information with the following keys:
                xl, xu, ytext, text, alpha
            defining the lower and upper aq values, the position of the text, the text,
            and the alpha of the range colouring.

        Raises
        ------
        ValueError
            If there are no analysis results.

        Returns
        -------
        fig,ax : Tuple(Matplotlib Figure, Matplotlib Axes)
            The figure and axes.

        """
        if not self.is_analysed:
            raise ValueError("Must be analysed before plotting")

        return self.beta_region_plot(
            self.aq_points, self.prob_models, beta_t, ranges, **kwargs
        )
