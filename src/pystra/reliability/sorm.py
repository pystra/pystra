#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

from .form import FORM
from .analysis import AnalysisObject
from scipy.stats import norm as normal
from ..errors import AnalysisError

__all__ = ["SORM"]


class SORM(AnalysisObject):
    r"""Second Order Reliability Method (SORM).

    Approximates the failure surface in standard normal space using a
    quadratic surface, improving on the linear FORM approximation. Two
    approaches are available:

    **Curve-fitting** (``fit_type='cf'``, default): computes the Hessian of
    the limit state function at the design point, extracts principal
    curvatures as eigenvalues, and applies the Breitung formula.

    **Point-fitting** (``fit_type='pf'``): locates fitting points directly
    on the failure surface on both sides of each principal axis using Newton
    iteration, then computes curvatures from their positions. This yields
    asymmetric curvatures (different on the positive and negative sides of
    each axis).

    Both methods report the Breitung [Breitung1984]_ and the
    Hohenbichler-Rackwitz [Hohenbichler1988]_ failure probabilities and
    generalised reliability indices.

    Parameters
    ----------
    stochastic_model : StochasticModel, optional
        The stochastic model with random variables and correlations.
    limit_state : LimitState, optional
        The limit state function.
    analysis_options : AnalysisOptions, optional
        Options controlling the analysis.
    form : FORM, optional
        A pre-computed FORM result. If ``None``, FORM is run automatically.
        SORM requires a successfully converged FORM analysis before running.

    Notes
    -----
    Result attributes are ``None`` before analysis and are cleared when a new
    run begins. ``results_valid`` becomes ``True`` only after fitting completes.

    Attributes
    ----------
    betaHL : float or ndarray
        Hasofer-Lind reliability index (from FORM).
    kappa : ndarray
        Principal curvatures. For curve-fitting, a 1-D array of length
        ``nrv - 1``. For point-fitting, the sorted average of the positive
        and negative curvatures.
    kappa_pf : ndarray or None
        Asymmetric curvatures from point-fitting, shape ``(2, nrv - 1)``
        with rows ``[kappa_minus, kappa_plus]``. ``None`` for curve-fitting.
    fit_type : str or None
        The fitting method used: ``'cf'``, ``'pf'``, or ``None`` if not
        yet run.
    pf2_breitung : float or ndarray
        Failure probability from the Breitung approximation.
    betag_breitung : float or ndarray
        Generalised reliability index from the Breitung approximation.
    pf2_breitung_m : float or ndarray
        Failure probability from the modified Breitung (Hohenbichler-Rackwitz).
    betag_breitung_m : float or ndarray
        Generalised reliability index from the modified Breitung.
    """

    def __init__(
        self, stochastic_model=None, limit_state=None, analysis_options=None, form=None
    ):
        """
        Class constructor
        """
        super().__init__(
            analysis_options=analysis_options,
            limit_state=limit_state,
            stochastic_model=stochastic_model,
        )

        # Has FORM already been run? If it exists it has, otherwise run it now
        if form is None:
            self.form = FORM(
                stochastic_model=self.model,
                limit_state=self.limitstate,
                analysis_options=self.options,
            )
            self.form.run()
        else:
            self.form = form

        self._reset_results()

    def _reset_results(self):
        """Invalidate previous SORM output before attempting another run."""
        self.results_valid = False
        self.betaHL = None
        self.kappa = None
        self.kappa_pf = None
        self.fit_type = None
        self.pf2_breitung = None
        self.betag_breitung = None
        self.pf2_breitung_m = None
        self.betag_breitung_m = None

    def _prepare_run(self, fit_type):
        """Require a valid FORM design point before preparing either fit."""
        self._reset_results()
        if not self.form.results_valid or not self.form.converged:
            raise AnalysisError("SORM requires a successfully converged FORM analysis")
        self.init_run()
        self.fit_type = fit_type

    def run(self, fit_type="cf"):
        """Run SORM analysis.

        Parameters
        ----------
        fit_type : {'cf', 'pf'}, optional
            Fitting method to use (default ``'cf'``):

            - ``'cf'`` — Curve-fitting via Hessian eigenvalues.
            - ``'pf'`` — Point-fitting via Newton iteration on the failure
              surface.

        Raises
        ------
        ValueError
            If *fit_type* is not ``'cf'`` or ``'pf'``.
        RuntimeError
            If the FORM analysis has not run or did not converge successfully.
        """
        if fit_type == "cf":
            self.run_curvefit()
        elif fit_type == "pf":
            self.run_pointfit()
        else:
            self._reset_results()
            raise ValueError(
                f"Unknown fit_type '{fit_type}'. Use 'cf' (curve-fitting) "
                f"or 'pf' (point-fitting)."
            )

    def run_curvefit(self):
        """Run curve-fitting; require a successfully converged FORM analysis."""
        self._prepare_run("cf")
        hess_G = self.compute_hessian()
        R1 = self.orthonormal_matrix()
        A = R1 @ hess_G @ R1.T / np.linalg.norm(self.form.gradient)
        kappa, _ = np.linalg.eig(A[:-1, :-1])
        kappa = np.real_if_close(kappa, tol=1e7)
        self.betaHL = self.form.beta
        self.kappa = np.sort(kappa)
        self.pf_breitung(self.betaHL, self.kappa)
        self.pf_breitung_m(self.betaHL, self.kappa)
        self.results_valid = True
        if self.options.get_print_output():
            self.show_results()

    def run_pointfit(self):
        """Run SORM analysis using point-fitting.

        Finds fitting points on the limit state surface on both the positive
        and negative sides of each principal axis in the rotated standard
        normal space.  Curvatures are computed from the positions of these
        points, producing asymmetric curvatures that are stored in
        :attr:`kappa_pf`.

        The generalized Breitung formula for asymmetric curvatures is:

        .. math::

            p_{f2} = \\Phi(-\\beta) \\prod_{i=1}^{n-1} \\frac{1}{2}
            \\left[ (1 + \\beta \\kappa_i^+)^{-1/2}
                  + (1 + \\beta \\kappa_i^-)^{-1/2} \\right]

        Notes
        -----
        Based on the point-fitting implementation contributed by
        Henry Nguyen (Monash University, PR #65).

        See Also
        --------
        run_curvefit : Alternative SORM approach using Hessian eigenvalues.

        Raises
        ------
        RuntimeError
            If the FORM analysis has not run or did not converge successfully,
            or a fitting point cannot be found.
        """
        self._prepare_run("pf")
        beta = self.form.get_beta()
        nrv = self.form.alpha.shape[1]
        R1 = self.orthonormal_matrix()
        marg = self.model.get_marginal_distributions()

        # Step coefficient controlling trial point distance from design point
        abs_beta = abs(beta)
        if abs_beta < 1:
            k = 1.0 / abs_beta
        elif abs_beta <= 3:
            k = 1.0
        else:
            k = 3.0 / abs_beta

        kappa_minus = np.zeros(nrv - 1)
        kappa_plus = np.zeros(nrv - 1)

        for i in range(nrv - 1):
            kappa_minus[i] = self._find_fitting_point(i, -1, beta, k, R1, marg)
            kappa_plus[i] = self._find_fitting_point(i, +1, beta, k, R1, marg)

        # Store results
        self.betaHL = self.form.beta
        self.kappa_pf = np.vstack([kappa_minus, kappa_plus])
        self.kappa = np.sort(0.5 * (kappa_minus + kappa_plus))

        # Compute failure probabilities
        self._pf_breitung_pf(beta, kappa_minus, kappa_plus)
        self._pf_breitung_m_pf(beta, kappa_minus, kappa_plus)

        self.results_valid = True
        if self.options.get_print_output():
            self.show_results()

    def _find_fitting_point(self, axis, sign, beta, k, R1, marg, max_iter=50, tol=1e-6):
        """Find a fitting point on the failure surface and return its curvature.

        Uses Newton iteration along the last axis of the rotated standard
        normal space to locate a point where :math:`G = 0`, keeping the
        coordinate on the fitting *axis* fixed.

        Parameters
        ----------
        axis : int
            Index of the rotated axis (``0`` to ``nrv - 2``).
        sign : {-1, +1}
            Side of the axis: ``-1`` for negative, ``+1`` for positive.
        beta : float
            Reliability index from FORM.
        k : float
            Step coefficient controlling trial point distance.
        R1 : ndarray
            Orthonormal rotation matrix, shape ``(nrv, nrv)``.
        marg : list
            Marginal distributions from the stochastic model.
        max_iter : int, optional
            Maximum Newton iterations (default 50).
        tol : float, optional
            Convergence tolerance on ``|G|`` (default ``1e-6``).

        Returns
        -------
        float
            Curvature :math:`a_i = 2 (u'_n - \\beta) / (u'_i)^2` for the
            given axis and side.

        Raises
        ------
        RuntimeError
            If Newton iteration does not converge within *max_iter* steps.
        """
        nrv = R1.shape[0]

        # Trial point in rotated space: u'[axis] = sign*k*beta, u'[-1] = beta
        u_prime = np.zeros(nrv)
        u_prime[axis] = sign * k * beta
        u_prime[-1] = beta

        G_val = None
        for iteration in range(max_iter):
            # Transform to standard normal space
            u = R1.T @ u_prime
            # Transform to physical space
            x = self.transform.u_to_x(u, marg)
            x_col = x[:, np.newaxis] if x.ndim == 1 else x

            # Evaluate LSF and gradient in u-space
            G, grad = self.evaluate_lsf(x_col, calc_gradient=True)
            G_val = np.squeeze(G)
            grad_u = np.squeeze(grad)

            if abs(G_val) < tol:
                break

            # Gradient in rotated space
            grad_rot = R1 @ grad_u

            # Newton update along the last axis (n-th direction)
            if abs(grad_rot[-1]) < 1e-12:
                raise AnalysisError(
                    f"Point-fitting: near-zero gradient component along the "
                    f"n-th axis at axis={axis}, sign={sign}. The limit state "
                    f"surface may be tangent to the search direction."
                )
            u_prime[-1] -= G_val / grad_rot[-1]
        else:
            raise AnalysisError(
                f"Point-fitting did not converge for axis={axis}, sign={sign} "
                f"after {max_iter} iterations (|G| = {abs(G_val):.2e})."
            )

        # Compute curvature from the converged fitting point
        u_prime_i = u_prime[axis]
        u_prime_n = u_prime[-1]

        if abs(u_prime_i) < 1e-12:
            return 0.0

        return 2.0 * (u_prime_n - beta) / (u_prime_i**2)

    def _pf_breitung_pf(self, beta, kappa_minus, kappa_plus):
        """Breitung formula for point-fitting with asymmetric curvatures.

        Parameters
        ----------
        beta : float
            Reliability index.
        kappa_minus : ndarray
            Curvatures on the negative side of each axis.
        kappa_plus : ndarray
            Curvatures on the positive side of each axis.
        """
        terms_plus = 1 + beta * kappa_plus
        terms_minus = 1 + beta * kappa_minus
        is_invalid = np.any(terms_plus <= 0) or np.any(terms_minus <= 0)

        if not is_invalid:
            self.pf2_breitung = float(
                normal.cdf(-beta)
                * np.prod(0.5 * (terms_plus ** (-0.5) + terms_minus ** (-0.5)))
            )
            self.betag_breitung = float(-normal.ppf(self.pf2_breitung))
        else:
            print("*** SORM Point-Fitting Breitung error, excessive curvatures")
            self.pf2_breitung = 0.0
            self.betag_breitung = 0.0

    def _pf_breitung_m_pf(self, beta, kappa_minus, kappa_plus):
        """Hohenbichler-Rackwitz modified Breitung for asymmetric curvatures.

        Parameters
        ----------
        beta : float
            Reliability index.
        kappa_minus : ndarray
            Curvatures on the negative side of each axis.
        kappa_plus : ndarray
            Curvatures on the positive side of each axis.
        """
        psi = normal.pdf(beta) / normal.cdf(-beta)
        terms_plus = 1 + psi * kappa_plus
        terms_minus = 1 + psi * kappa_minus
        is_invalid = np.any(terms_plus <= 0) or np.any(terms_minus <= 0)

        if not is_invalid:
            self.pf2_breitung_m = float(
                normal.cdf(-beta)
                * np.prod(0.5 * (terms_plus ** (-0.5) + terms_minus ** (-0.5)))
            )
            self.betag_breitung_m = float(-normal.ppf(self.pf2_breitung_m))
        else:
            print("*** SORM Point-Fitting Breitung Modified error")
            self.pf2_breitung_m = 0.0
            self.betag_breitung_m = 0.0

    def pf_breitung(self, beta, kappa):
        """
        Calculates the probability of failure and generalized reliability
        index using [Breitung1984]_ formula. This formula is good for higher
        values of beta.
        """
        is_invalid = np.any(kappa < -1 / beta)
        if not is_invalid:
            self.pf2_breitung = float(
                normal.cdf(-beta) * np.prod((1 + beta * kappa) ** (-0.5))
            )
            self.betag_breitung = float(-normal.ppf(self.pf2_breitung))
        else:
            print("*** SORM Breitung error, excessive curavtures")
            self.pf2_breitung = 0.0
            self.betag_breitung = 0.0

    def pf_breitung_m(self, beta, kappa):
        """
        Calculates the probability of failure and generalized reliability
        index using Brietung's formula ([Breitung1984]_) as modified by Hohenbichler and
        Rackwitz [Hohenbichler1988]_. This formula is better for lower values of beta.

        """
        k = normal.pdf(beta) / normal.cdf(-beta)
        is_invalid = np.any(kappa < -1 / k)
        if not is_invalid:
            self.pf2_breitung_m = float(
                normal.cdf(-beta) * np.prod((1 + k * kappa) ** (-0.5))
            )
            self.betag_breitung_m = float(-normal.ppf(self.pf2_breitung_m))
        else:
            print("*** SORM Breitung Modified error")
            self.pf2_breitung_m = 0.0
            self.betag_breitung_m = 0.0

    def show_results(self):
        """Print a compact summary of the SORM results."""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        n_hyphen = self.N_HYPH
        fit_label = " (Point-Fitting)" if self.fit_type == "pf" else ""
        print("")
        print("=" * n_hyphen)
        print("")
        print(f"RESULTS FROM RUNNING SECOND ORDER RELIABILITY METHOD{fit_label}")
        print("")
        print("Generalized reliability index: ", self.betag_breitung)
        print("Probability of failure:        ", self.pf2_breitung)
        print("")
        if self.fit_type == "pf" and self.kappa_pf is not None:
            for i in range(self.kappa_pf.shape[1]):
                print(
                    f"Curvature {i+1}:  kappa- = {self.kappa_pf[0, i]:+.6f}"
                    f"  kappa+ = {self.kappa_pf[1, i]:+.6f}"
                )
        else:
            for i, k in enumerate(self.kappa):
                print(f"Curvature {i+1}: {k}")
        print("=" * n_hyphen)
        print("")

    def show_detailed_output(self):
        """Print detailed FORM/SORM comparison to the console."""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        names = self.model.get_variables().keys()
        consts = self.model.get_constants()
        u_star = self.form.get_design_point()
        x_star = self.form.get_design_point(uspace=False)
        alpha = self.form.get_alpha()
        betaHL = self.form.get_beta()
        pfFORM = self.form.get_failure()

        fit_label = " (Point-Fitting)" if self.fit_type == "pf" else ""
        n_hyphen = self.N_HYPH
        print("")
        print("=" * n_hyphen)
        print(f"FORM/SORM{fit_label}")
        print("=" * n_hyphen)
        print("{:15s} \t\t {:1.10e}".format("Pf FORM", pfFORM))
        print("{:15s} \t\t {:1.10e}".format("Pf SORM Breitung", self.pf2_breitung))
        print("{:15s} \t {:1.10e}".format("Pf SORM Breitung HR", self.pf2_breitung_m))
        print("{:15s} \t\t {:2.10f}".format("Beta_HL", betaHL))
        print("{:15s} \t\t {:2.10f}".format("Beta_G Breitung", self.betag_breitung))
        print(
            "{:15s} \t\t {:2.10f}".format("Beta_G Breitung HR", self.betag_breitung_m)
        )
        print(
            "{:15s} \t\t {:d}".format(
                "Model Evaluations", self.model.get_call_function()
            )
        )

        print("-" * n_hyphen)
        if self.fit_type == "pf" and self.kappa_pf is not None:
            for i in range(self.kappa_pf.shape[1]):
                print(
                    f"Curvature {i+1}:  kappa- = {self.kappa_pf[0, i]:+.6f}"
                    f"  kappa+ = {self.kappa_pf[1, i]:+.6f}"
                )
        else:
            for i, k in enumerate(self.kappa):
                print(f"Curvature {i+1}: {k}")

        print("-" * n_hyphen)
        print(
            "{:10s} \t {:>9s} \t {:>12s} \t {:>9s}".format(
                "Variable", "U_star", "X_star", "alpha"
            )
        )
        for i, name in enumerate(names):
            print(
                "{:10s} \t {: 5.6f} \t {:12.6f} \t {:+5.6f}".format(
                    name, u_star[i], x_star[i], alpha[i]
                )
            )
        for name, val in consts.items():
            print(f"{name:10s} \t {'---':>9s} \t {val:12.6f} \t {'---':>9s}")
        print("=" * n_hyphen)
        print("")

    def compute_hessian(self, diff_type=None):
        """
        Computes the matrix of second derivatives using forward finite
        difference, using the evaluation of the gradient already done
        for FORM, at the design point

        Could use numdifftools as external library instead

        """

        h = 1 / self.options.ffdpara
        nrv = self.form.alpha.shape[1]
        hess_G = np.zeros((nrv, nrv))

        if diff_type is None:
            # Differentiation based on the gradients
            x0 = self.form.get_design_point(uspace=False)
            _, grad_g0 = self.evaluate_lsf(x0[:, np.newaxis], calc_gradient=True)
            u0 = self.form.get_design_point()
            for i in range(nrv):
                u1 = np.copy(u0)
                u1[i] += h
                x1 = self.transform.u_to_x(u1, self.model.get_marginal_distributions())
                _, grad_g1 = self.evaluate_lsf(x1[:, np.newaxis], calc_gradient=True)
                hess_G[:, i] = ((grad_g1 - grad_g0) / h).reshape(nrv)

        else:
            # FERUM-implementation using a mix of central and foward diffs
            # It would be good if u_to_x could take an nvr x nx matrix and
            # return the corresponding x-matrix. This would make it easier to
            # add more numerical differentiation schemes.
            u0 = self.form.get_design_point()

            all_x_plus = np.zeros((nrv, nrv))
            all_x_minus = np.zeros((nrv, nrv))
            all_x_both = np.zeros((nrv, int(nrv * (nrv - 1) / 2)))

            marg = self.model.get_marginal_distributions()
            for i in range(nrv):
                # Plus perturbation and transformation
                u_plus = np.copy(u0)
                u_plus[i] += h
                x_plus = self.transform.u_to_x(u_plus, marg)
                all_x_plus[:, i] = x_plus

                # Minus perturbation and transformation
                u_minus = np.copy(u0)
                u_minus[i] -= h
                x_minus = self.transform.u_to_x(u_minus, marg)
                all_x_minus[:, i] = x_minus

                for j in range(i):
                    # Mixed perturbation and transformation
                    u_both = np.copy(u_plus)
                    u_both[j] += h
                    x_both = self.transform.u_to_x(
                        u_both, self.model.get_marginal_distributions()
                    )
                    all_x_both[:, int((i - 1) * (i) / 2) + j] = x_both

            # Assemble all x-space vecs, solve for G, then separate
            all_x = np.concatenate((all_x_plus, all_x_minus, all_x_both), axis=1)
            all_G, _ = self.evaluate_lsf(all_x, calc_gradient=False)
            all_G = all_G.squeeze()
            all_G_plus = all_G[:nrv]
            all_G_minus = all_G[nrv : 2 * nrv]
            all_G_both = all_G[2 * nrv : :]
            G = self.form.G

            # Now use finite differences to estimate hessian
            for i in range(nrv):
                # Second-order central difference
                hess_G[i, i] = (all_G_plus[i] - 2 * G + all_G_minus[i]) / h**2
                for j in range(i):
                    # Second order forward difference
                    hess_G[i, j] = (
                        all_G_both[int((i - 1) * (i) / 2) + j]
                        - all_G_plus[j]
                        - all_G_plus[i]
                        + G
                    ) / h**2
                    hess_G[j, i] = hess_G[i, j]

        return hess_G

    def evaluate_lsf(self, x, calc_gradient=False, u_space=True):
        """
        For use in computing the Hessian without altering the FORM object.
        Considers the coord transform so the limit state function is evaluated
        in physical coordinates, but gradient returned in u-space.

        This code already in FORM, and a more integrated approach would put
        this in a base class for common use.

        """
        G, grad = 0, 0
        x0 = np.copy(x)  # avoid modifying argument in func calls below

        if calc_gradient:
            G, grad = self.limitstate.evaluate_lsf(x0, self.model, self.options)
            grad = np.transpose(grad)
            if u_space:
                marg = self.model.get_marginal_distributions()
                u = self.transform.x_to_u(x0, marg)
                J_u_x = self.transform.jacobian(u, x0, marg)
                J_x_u = np.linalg.inv(J_u_x)
                grad = np.dot(grad, J_x_u)
        else:
            G, _ = self.limitstate.evaluate_lsf(x0, self.model, self.options)

        return G, grad

    def orthonormal_matrix(self):
        """
        Computes the rotation matrix of the standard normal coordinate
        space where the design point is located at Beta along the last
        axis.
        """
        alpha = self.form.alpha.squeeze()
        nrv = len(alpha)
        # Assemble A per theory
        A = np.eye(nrv).copy()
        A[-1, :] = alpha
        # Now rotate 270 degrees anticlockwise as MGS expects the reference
        # vector in the first column
        A = np.rot90(A, k=3)
        # Orthoganalize
        Q = self.gram_schmidt(A)
        # And undo this rotation with a final 90 dgree rotation as order of
        # the column vector entries is not relevant
        R1 = np.rot90(Q)
        return R1

    def gram_schmidt(self, A):
        """
        Creates an orthonormal matrix using the modified Gram-Schmidt process.
        Note that QR decomposition doesn't work for this application; while
        it does return an orthonormal matrix, the signs are different
        to the modified Gram Schmidt. The signs should be arbitrary, but the
        resulting rotation matrix does care cabout the signs of the Q, since
        it is based on the correct direction of the beta vector [Madsen1986]
        """

        A = np.asarray(A, dtype="float")
        nvr = A.shape[0]
        Q = np.zeros(A.shape)
        for j in range(nvr):
            q = A[:, j]
            for i in range(j):
                rij = np.dot(q, Q[:, i])
                q = q - rij * Q[:, i]
            rjj = np.linalg.norm(q, ord=2)
            if np.isclose(rjj, 0.0):
                raise ValueError("Singular rotation matrix")
            else:
                Q[:, j] = q / rjj
        return Q
