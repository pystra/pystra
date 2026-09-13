"""Sensitivity analysis of the reliability index.

This module computes the sensitivity of the FORM reliability index
with respect to each distribution parameter declared by
:attr:`Distribution.sensitivity_params` (by default the mean and
standard deviation, but subclasses may add shape parameters, etc.).

Two methods are available, selected by the ``method`` argument of
:class:`SensitivityAnalysis`:

- **Finite difference** (``method="numerical"``, default): perturbs each
  parameter and re-runs FORM.  Simple but expensive.
- **Closed-form** (``method="closed_form"``): post-processes a single FORM
  run using the Cholesky-differentiation approach of
  Bourinet (2017) [Bourinet2017]_.  Also computes correlation
  sensitivities.  Faster and more accurate, especially when the
  number of variables is large.
"""

from typing import Literal
import copy
import math
from numbers import Real

import numpy as np

import pystra as _pystra
from .form import FORM
from ..errors import AnalysisError, ModelError
from .analysis import _check_on_failure
from ..model import LimitState, StochasticModel
from ..options import FORMOptions
from ..results import SensitivityResult
from .._numerics.cholesky_sensitivity import (
    cholesky_with_derivative,
    inverse_cholesky_gradient,
)
from .._numerics.integration import zi_and_xi, drho_drho0, drho0_dtheta

__all__ = ["SensitivityAnalysis"]


class SensitivityAnalysis:
    r"""Sensitivity analysis for the FORM reliability index.

    Computes :math:`\partial\beta / \partial\theta` for each distribution
    parameter :math:`\theta` declared by
    :attr:`Distribution.sensitivity_params` (by default, mean and standard
    deviation; subclasses may add shape parameters, etc.).

    Two algorithms are available, selected by ``method``:

    - ``"numerical"`` — forward finite differences (default).
    - ``"closed_form"`` — closed-form post-processing of a single FORM run
      using the approach of Bourinet (2017) [Bourinet2017]_. This also
      returns sensitivities to correlation coefficients.

    Parameters
    ----------
    model : StochasticModel
        The stochastic model (distributions + correlation).
    limit_state : LimitState
        The limit state function definition.
    options : FORMOptions, optional
        Settings for every FORM run.
    method : {"numerical", "closed_form"}, default "numerical"
        Forward finite differences, in which each parameter is perturbed by
        ``delta`` times its standard deviation and FORM is rerun, or the
        closed form, which post-processes one FORM run.
    delta : float, default 0.01
        Relative perturbation of the numerical method; the closed form does
        not use it.
    on_failure : {"raise", "return"}, default "raise"
        If a FORM analysis does not converge, raise :class:`~pystra.AnalysisError`, which carries
        the unconverged record in ``.result``, or return that record.
    """

    def __init__(
        self,
        model: "_pystra.StochasticModel",
        limit_state: "_pystra.LimitState",
        *,
        options: "_pystra.FORMOptions | None" = None,
        method: Literal["numerical", "closed_form"] = "numerical",
        delta: float = 0.01,
        on_failure: Literal["raise", "return"] = "raise",
    ) -> None:
        if not isinstance(model, StochasticModel):
            raise TypeError("SensitivityAnalysis requires a StochasticModel")
        if not isinstance(limit_state, LimitState):
            raise TypeError("SensitivityAnalysis requires a LimitState")
        if options is None:
            options = FORMOptions()
        elif not isinstance(options, FORMOptions):
            raise TypeError(
                f"SensitivityAnalysis takes FORMOptions, not {type(options).__name__}"
            )
        if method not in ("numerical", "closed_form"):
            raise ModelError("method must be 'numerical' or 'closed_form'")
        if (
            isinstance(delta, bool)
            or not isinstance(delta, Real)
            or not math.isfinite(delta)
            or delta <= 0
        ):
            raise ModelError("delta must be a finite positive number")
        if method == "closed_form" and delta != 0.01:
            raise ModelError("The closed form does not use delta")
        self.model = model
        self.limit_state = limit_state
        self.options = options
        self.method = method
        self.delta = delta
        self.on_failure = _check_on_failure(on_failure)

    def __repr__(self) -> str:
        names = tuple(self.model.get_variables())
        return (
            f"SensitivityAnalysis(variables={names!r}, method={self.method!r}, "
            f"options={self.options!r})"
        )

    def run(self) -> "_pystra.SensitivityResult":
        r"""Run the sensitivity analysis.

        Returns
        -------
        SensitivityResult
            ``marginal`` maps each variable name to the derivatives with
            respect to its distribution parameters, keyed by the names in its
            :attr:`sensitivity_params` (typically ``"mean"`` and ``"std"``,
            and possibly ``"shape"``). The closed form also gives
            ``correlation``, a symmetric *n × n* array whose element *(i, j)*
            is :math:`\partial\beta / \partial\rho_{ij}`, with a zero diagonal.
        """
        self._evaluations, self._converged = 0, True
        self._diagnostics = {}
        numerical = self.method == "numerical"
        if numerical:
            base, marginal = self._numerical_sens(self.delta)
            correlation = None
        else:
            base, marginal, correlation = self._cf_sens()
        ok = self._converged
        result = SensitivityResult(
            method="SensitivityAnalysis",
            status="converged" if self._converged else "not_converged",
            message=(
                "Every FORM analysis converged"
                if self._converged
                else "A FORM analysis did not converge"
            ),
            n_limit_state_evaluations=self._evaluations,
            variable_names=base.variable_names,
            failure_probability=base.failure_probability if ok else None,
            beta=base.beta if ok else None,
            form=base,
            approach=self.method,
            marginal=(
                {
                    name: {param: float(value) for param, value in params.items()}
                    for name, params in marginal.items()
                }
                if ok
                else {}
            ),
            correlation=correlation if ok else None,
            delta=self.delta if numerical else None,
            options=self.options,
            diagnostics=self._diagnostics,
        )
        if not ok and self.on_failure == "raise":
            raise AnalysisError(result.message, result)
        return result

    def _form(self, model, **context):
        """Run FORM on *model*, recording its evaluations and convergence."""
        form = FORM(model, self.limit_state, options=self.options, on_failure="return")
        result = form.run()
        self._evaluations += result.n_limit_state_evaluations
        self._converged = self._converged and result.converged
        if not result.converged:
            self._diagnostics = {"failed_form": result, **context}
        return form, result

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
        variables = self.model.get_variables()
        names = list(variables.keys())

        # Build result dict with per-variable parameter keys
        sensitivities = {}
        for name in names:
            dist = variables[name]
            sensitivities[name] = {p: 0.0 for p in dist.sensitivity_params}

        # Get the base result
        form, base = self._form(self.model, phase="baseline")
        if not base.converged:
            return base, {}
        beta0 = form._beta

        for name in names:
            dist = variables[name]
            for param, val in dist.sensitivity_params.items():
                model1 = copy.deepcopy(self.model)
                dist1 = model1.variable(name)

                # Perturb and replace using with_parameters
                h = delta * dist1.std
                new_dist = dist1.with_parameters(
                    **{**dist1.sensitivity_params, param: val + h}, start_point=None
                )
                # Replace in both variables dict and _marg list
                marg_idx = list(model1.variables.keys()).index(name)
                model1.variables[name] = new_dist
                model1._marg[marg_idx] = new_dist
                delta_actual = new_dist.sensitivity_params[param] - val

                # Run FORM with perturbed model
                form, perturbed = self._form(
                    model1,
                    phase="perturbation",
                    variable=name,
                    parameter=param,
                    step=delta_actual,
                )
                if not perturbed.converged:
                    return base, {}
                beta1 = form._beta
                sensitivities[name][param] = (beta1 - beta0) / delta_actual

        return base, sensitivities

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
        if self.model.get_copula() is not None or self.options.transform in (
            "nataf",
            "rosenblatt",
        ):
            raise ValueError(
                "Closed-form sensitivities assume legacy physical Pearson input; use numerical=True for explicit copulas"
            )
        # 1. Run FORM
        form, base = self._form(self.model, phase="baseline")
        if not base.converged:
            return base, {}, None

        # Extract converged quantities
        alpha = form._alpha[0]  # shape (nrv,)
        u_star = form._u  # shape (nrv,)
        x_star = form._design_point_x()  # shape (nrv,)

        marg = self.model.get_marginal_distributions()
        nrv = len(marg)
        R = self.model.get_correlation()  # physical correlation
        Ro = self.model.get_modified_correlation()  # modified (Nataf) correlation

        L0 = form.transform.inv_T  # Cholesky factor, shape (n,n)
        L0_inv = form.transform.T  # its inverse

        z_star = L0 @ u_star  # correlated std normal at design point

        variables = self.model.get_variables()
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
                nIP = self._select_n_ip(rho)
                grid = zi_and_xi(marg[i], marg[j], 6, nIP)
                quad_grids[(i, j)] = grid

        for var_k, name_k in enumerate(names):
            dist_k = marg[var_k]

            for param in dist_k.sensitivity_params:
                # --- First term: αᵀ L₀⁻¹ (∂z/∂θ_k) ---
                # ∂z_i/∂θ_k is nonzero only for i == var_k  (Eq. 22)
                dF = dist_k.cdf_gradient(x_star[var_k])
                phi_z = dist_k.std_normal.pdf(z_star[var_k])
                dz_dtheta = np.zeros(nrv)
                if phi_z > 1e-300:
                    dz_dtheta[var_k] = dF[param] / phi_z

                term1 = alpha @ (L0_inv @ dz_dtheta)

                # --- Second term: αᵀ (∂L₀⁻¹/∂θ_k) z ---
                # Need ∂R₀/∂θ_k, then Cholesky diff → ∂L₀/∂θ_k → ∂L₀⁻¹/∂θ_k
                dR0_dtheta = self._compute_d_r0_dtheta(
                    marg, nrv, Ro, R, quad_grids, var_k, param
                )

                _, dL0 = cholesky_with_derivative(Ro, dR0_dtheta)
                dL0_inv = inverse_cholesky_gradient(L0, dL0)

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

                drho_val = drho_drho0(rho0_ij, marg[i], marg[j], *grid)
                # ∂ρ₀,ij/∂ρ_ij = (∂ρ_ij/∂ρ₀,ij)⁻¹  (Eq. 20)
                drho0_drho = 1.0 / drho_val if abs(drho_val) > 1e-300 else 0.0

                # Build ∂R₀/∂ρ_ij matrix
                dR0 = np.zeros((nrv, nrv))
                dR0[i, j] = drho0_drho
                dR0[j, i] = drho0_drho

                # Cholesky diff → ∂L₀⁻¹/∂ρ_ij
                _, dL0 = cholesky_with_derivative(Ro, dR0)
                dL0_inv = inverse_cholesky_gradient(L0, dL0)

                # ∂β/∂ρ_ij = αᵀ (∂L₀⁻¹/∂ρ_ij) z  (first term vanishes)
                dbeta = float(alpha @ (dL0_inv @ z_star))
                corr_sens[i, j] = dbeta
                corr_sens[j, i] = dbeta

        return base, marginal_sens, corr_sens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_d_r0_dtheta(self, marg, nrv, Ro, R, quad_grids, var_k, param):
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

                val = drho0_dtheta(rho0_ij, marg[i], marg[j], *grid, vi, param)
                dR0[i, j] = val
                dR0[j, i] = val

        return dR0

    @staticmethod
    def _select_n_ip(rho):
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
