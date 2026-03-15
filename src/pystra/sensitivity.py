#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitivity analysis of the reliability index.

This module computes the sensitivity of the FORM reliability index
with respect to each distribution parameter declared by
:attr:`Distribution.sensitivity_params` (by default the mean and
standard deviation, but subclasses may add shape parameters, etc.).

Two methods are available, selected via the ``numerical`` flag of
:meth:`SensitivityAnalysis.run`:

- **Finite difference** (``numerical=True``, default): perturbs each
  parameter and re-runs FORM.  Simple but expensive.
- **Closed-form** (``numerical=False``): post-processes a single FORM
  run using the Cholesky-differentiation approach of
  Bourinet (2017) [Bourinet2017]_.  Also computes correlation
  sensitivities.  Faster and more accurate, especially when the
  number of variables is large.
"""

from .form import Form
from .analysis import AnalysisOptions
from .cholesky_sensitivity import cholesky_with_derivative, dinvL0_dtheta
from .integration import zi_and_xi, drho_drho0, drho0_dtheta
import copy
import numpy as np


class SensitivityAnalysis:
    r"""Sensitivity analysis for the FORM reliability index.

    Computes :math:`\partial\beta / \partial\theta` for each distribution
    parameter :math:`\theta` declared by
    :attr:`Distribution.sensitivity_params` (by default, mean and standard
    deviation; subclasses may add shape parameters, etc.).

    Two algorithms are available, selected by the ``numerical`` argument
    of :meth:`run`:

    - ``numerical=True``  — forward finite differences (default).
    - ``numerical=False`` — closed-form post-processing of a single FORM
      run using the approach of Bourinet (2017) [Bourinet2017]_.  This
      also returns sensitivities to correlation coefficients.

    Parameters
    ----------
    limit_state : LimitState
        The limit state function definition.
    stochastic_model : StochasticModel
        The stochastic model (distributions + correlation).
    analysis_options : AnalysisOptions, optional
        Options forwarded to each FORM run.  Defaults are used if
        ``None``.
    """

    def __init__(self, limit_state, stochastic_model, analysis_options=None):
        self.limitstate = limit_state
        self.model = stochastic_model

        # Options for the calculation
        if analysis_options is None:
            self.options = AnalysisOptions()
        else:
            self.options = analysis_options

    def run(self, numerical=True, delta=0.01):
        r"""Run FORM-based sensitivity analysis.

        Parameters
        ----------
        numerical : bool, optional
            If ``True`` (default), use forward finite differences: each
            parameter is perturbed by ``delta * stdv`` and FORM is
            re-run.  If ``False``, use the closed-form approach of
            Bourinet (2017) which post-processes a single FORM run.
        delta : float, optional
            Relative perturbation size for the finite-difference method
            (default 0.01, i.e. 1 %).  Ignored when ``numerical=False``.

        Returns
        -------
        dict
            When ``numerical=True``:
                ``{variable_name: {param: dβ/dθ, ...}}``.
                The parameter keys come from each distribution's
                :attr:`sensitivity_params` (typically ``"mean"`` and
                ``"std"``, but may include ``"shape"`` etc.).

            When ``numerical=False``:
                ``{"marginal": {...}, "correlation": ndarray}``.

                The ``"correlation"`` entry is a symmetric *n × n* array
                where element *(i, j)* is :math:`\partial\beta /
                \partial\rho_{ij}`.  Diagonal entries are zero.
        """
        if numerical:
            return self._numerical_sens(delta)
        else:
            return self._cf_sens()

    def run_form(self, numerical=True, delta=0.01):
        """Alias for :meth:`run` (backwards compatibility)."""
        return self.run(numerical=numerical, delta=delta)

    def summary(self, result):
        """Return a pandas DataFrame summarising sensitivity results.

        Converts the nested dict returned by :meth:`run` into a tidy
        DataFrame for convenient display in notebooks.

        Parameters
        ----------
        result : dict
            The result dict from :meth:`run` (either FD or CF format).

        Returns
        -------
        pandas.DataFrame
            Columns: ``Variable``, ``Parameter``, ``∂β/∂θ``.
        """
        import pandas as pd

        # Handle both FD result (flat) and CF result (nested with "marginal")
        marginal = result.get("marginal", result)
        rows = []
        for var_name, params in marginal.items():
            for param, value in params.items():
                rows.append({
                    "Variable": var_name,
                    "Parameter": param,
                    "∂β/∂θ": value,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private: finite-difference sensitivities
    # ------------------------------------------------------------------
    def _numerical_sens(self, delta):
        r"""Forward finite-difference sensitivity analysis.

        For each random variable, every parameter declared by
        :attr:`sensitivity_params` is perturbed by ``delta * stdv``
        and a new FORM analysis is executed.  The sensitivity is the
        finite-difference approximation
        :math:`(\beta_1 - \beta_0) / \Delta\theta`.
        """
        variables = self.model.getVariables()
        names = list(variables.keys())

        # Build result dict with per-variable parameter keys
        sensitivities = {}
        for name in names:
            dist = variables[name]
            sensitivities[name] = {p: 0.0 for p in dist.sensitivity_params}

        # Get the base result
        form = Form(stochastic_model=self.model, limit_state=self.limitstate)
        form.run()
        beta0 = form.getBeta()

        for name in names:
            dist = variables[name]
            for param, val in dist.sensitivity_params.items():
                model1 = copy.deepcopy(self.model)
                dist1 = model1.getVariable(name)

                # Perturb and replace using _make_copy
                h = delta * dist1.stdv
                new_dist = dist1._make_copy(**{param: val + h})
                # Replace in both variables dict and _marg list
                marg_idx = list(model1.variables.keys()).index(name)
                model1.variables[name] = new_dist
                model1._marg[marg_idx] = new_dist
                delta_actual = new_dist.sensitivity_params[param] - val

                # Run FORM with perturbed model
                form = Form(
                    stochastic_model=model1,
                    limit_state=self.limitstate,
                )
                form.run()
                beta1 = form.getBeta()
                sensitivities[name][param] = (beta1 - beta0) / delta_actual

        return sensitivities

    # ------------------------------------------------------------------
    # Private: closed-form (Bourinet 2017) sensitivities
    # ------------------------------------------------------------------
    def _cf_sens(self):
        r"""Closed-form sensitivity analysis (Bourinet 2017).

        Runs a single FORM analysis then post-processes the converged
        design point to obtain sensitivities of :math:`\beta` to:

        - marginal distribution parameters (as declared by each
          distribution's :attr:`sensitivity_params`),
        - correlation coefficients.

        Uses the Cholesky-differentiation algorithm from the Appendix
        of Bourinet (2017) and Eqs. (17)–(25) for the derivative
        integrals.  No additional FORM runs are required.
        """
        # 1. Run FORM
        form = Form(
            stochastic_model=self.model,
            limit_state=self.limitstate,
            analysis_options=self.options,
        )
        form.run()

        # Extract converged quantities
        alpha = form.getAlpha()              # shape (nrv,)
        u_star = form.getDesignPoint()       # shape (nrv,)
        x_star = form.getDesignPoint(uspace=False)  # shape (nrv,)

        marg = self.model.getMarginalDistributions()
        nrv = len(marg)
        R = self.model.getCorrelation()      # physical correlation
        Ro = self.model.getModifiedCorrelation()  # modified (Nataf) correlation

        L0 = form.transform.inv_T            # Cholesky factor, shape (n,n)
        L0_inv = form.transform.T            # its inverse

        z_star = L0 @ u_star                 # correlated std normal at design point

        variables = self.model.getVariables()
        names = list(variables.keys())

        # ------------------------------------------------------------------
        # Marginal parameter sensitivities  (Eq. 17)
        # ------------------------------------------------------------------
        marginal_sens = {}
        for name in names:
            dist = variables[name]
            marginal_sens[name] = {p: 0.0 for p in dist.sensitivity_params}

        # Pre-compute quadrature grids for each pair (needed for second term)
        quad_grids = {}
        for i in range(nrv):
            for j in range(i):
                rho = R[i, j]
                nIP = self._select_nIP(rho)
                grid = zi_and_xi(marg[i], marg[j], 6, nIP)
                quad_grids[(i, j)] = grid

        for var_k, name_k in enumerate(names):
            dist_k = marg[var_k]

            for param in dist_k.sensitivity_params:
                # --- First term: αᵀ L₀⁻¹ (∂z/∂θ_k) ---
                # ∂z_i/∂θ_k is nonzero only for i == var_k  (Eq. 22)
                dF = dist_k.dF_dtheta(x_star[var_k])
                phi_z = dist_k.std_normal.pdf(z_star[var_k])
                dz_dtheta = np.zeros(nrv)
                if phi_z > 1e-300:
                    dz_dtheta[var_k] = dF[param] / phi_z

                term1 = alpha @ (L0_inv @ dz_dtheta)

                # --- Second term: αᵀ (∂L₀⁻¹/∂θ_k) z ---
                # Need ∂R₀/∂θ_k, then Cholesky diff → ∂L₀/∂θ_k → ∂L₀⁻¹/∂θ_k
                dR0_dtheta = self._compute_dR0_dtheta(
                    marg, nrv, Ro, R, quad_grids, var_k, param
                )

                _, dL0 = cholesky_with_derivative(Ro, dR0_dtheta)
                dL0_inv = dinvL0_dtheta(L0, dL0)

                term2 = alpha @ (dL0_inv @ z_star)

                marginal_sens[name_k][param] = float(term1 + term2)

        # ------------------------------------------------------------------
        # Correlation sensitivities  (Eq. 18)
        # ------------------------------------------------------------------
        corr_sens = np.zeros((nrv, nrv))

        for i in range(nrv):
            for j in range(i):
                # ∂R₀/∂ρ_ij  (Eq. 20)
                grid = quad_grids[(i, j)]
                rho0_ij = Ro[i, j]

                drho_val = drho_drho0(
                    rho0_ij, marg[i], marg[j], *grid
                )
                # ∂ρ₀,ij/∂ρ_ij = (∂ρ_ij/∂ρ₀,ij)⁻¹  (Eq. 20)
                drho0_drho = 1.0 / drho_val if abs(drho_val) > 1e-300 else 0.0

                # Build ∂R₀/∂ρ_ij matrix
                dR0 = np.zeros((nrv, nrv))
                dR0[i, j] = drho0_drho
                dR0[j, i] = drho0_drho

                # Cholesky diff → ∂L₀⁻¹/∂ρ_ij
                _, dL0 = cholesky_with_derivative(Ro, dR0)
                dL0_inv = dinvL0_dtheta(L0, dL0)

                # ∂β/∂ρ_ij = αᵀ (∂L₀⁻¹/∂ρ_ij) z  (first term vanishes)
                dbeta = float(alpha @ (dL0_inv @ z_star))
                corr_sens[i, j] = dbeta
                corr_sens[j, i] = dbeta

        return {"marginal": marginal_sens, "correlation": corr_sens}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_dR0_dtheta(self, marg, nrv, Ro, R, quad_grids, var_k, param):
        r"""Build ∂R₀/∂θ_k for a marginal distribution parameter.

        For each pair (i, j), computes ``∂ρ₀,ij/∂θ_k`` using
        :func:`drho0_dtheta` when the pair involves variable *var_k*
        and the distributions are not both normal (where ρ₀ = ρ and is
        independent of marginal parameters).

        Parameters
        ----------
        marg : list of Distribution
            Marginal distributions.
        nrv : int
            Number of random variables.
        Ro : ndarray
            Modified correlation matrix.
        R : ndarray
            Physical correlation matrix.
        quad_grids : dict
            Pre-computed quadrature grids keyed by ``(i, j)`` with i > j.
        var_k : int
            Index of the variable whose parameter is being differentiated.
        param : str
            Parameter name (a key of the variable's
            :attr:`sensitivity_params`).

        Returns
        -------
        ndarray, shape (nrv, nrv)
            Symmetric matrix ``∂R₀/∂θ_k``.
        """
        dR0 = np.zeros((nrv, nrv))

        for i in range(nrv):
            for j in range(i):
                # Only nonzero if var_k is one of the pair members
                if var_k != i and var_k != j:
                    continue

                # If ρ₀,ij = ρ_ij (e.g. both normal), then ∂ρ₀/∂θ_k = 0
                if abs(Ro[i, j] - R[i, j]) < 1e-12 and abs(R[i, j]) < 1e-12:
                    continue

                grid = quad_grids[(i, j)]
                rho0_ij = Ro[i, j]

                # var_idx: which of the pair (0=margi, 1=margj) is var_k
                if var_k == i:
                    vi = 0
                else:
                    vi = 1

                val = drho0_dtheta(
                    rho0_ij, marg[i], marg[j], *grid, vi, param
                )
                dR0[i, j] = val
                dR0[j, i] = val

        return dR0

    @staticmethod
    def _select_nIP(rho):
        """Select the number of integration points based on |ρ|."""
        rho_abs = abs(rho)
        if rho_abs > 0.9995:
            return 1024
        elif rho_abs > 0.998:
            return 512
        elif rho_abs > 0.992:
            return 256
        elif rho_abs > 0.97:
            return 128
        elif rho_abs > 0.9:
            return 64
        else:
            return 32
