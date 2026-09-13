#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

from .form import FORM
from .analysis import AnalysisObject, _check_on_failure
from ._form_reuse import _check_form, _check_coordinates, _FORMReuse
from scipy.stats import norm as normal
from scipy.special import log_ndtr, ndtri_exp
from ..errors import AnalysisError, ModelError
from ..options import FORMOptions, SORMOptions
from ..results import FORMResult, SORMResult

__all__ = ["SORM"]


class SORM(_FORMReuse, AnalysisObject):
    r"""Second Order Reliability Method (SORM).

    Approximates the failure surface in standard normal space using a
    quadratic surface, improving on the linear FORM approximation. Two
    approaches are available:

    **Curve-fitting** (``fit="curve"``, default): computes the Hessian of
    the limit state function at the design point, extracts principal
    curvatures as eigenvalues, and applies the Breitung formula.

    **Point-fitting** (``fit="point"``): locates fitting points directly
    on the failure surface on both sides of each principal axis using Newton
    iteration, then computes curvatures from their positions. This yields
    asymmetric curvatures (different on the positive and negative sides of
    each axis).

    Both methods report the Breitung [Breitung1984]_ and the
    Hohenbichler-Rackwitz [Hohenbichler1988]_ failure probabilities and
    generalised reliability indices.

    Parameters
    ----------
    model : StochasticModel
        The stochastic model with random variables and correlations.
    limit_state : LimitState
        The limit state function.
    options : SORMOptions, optional
        The fit, the reported formula and the Hessian step.
    form : FORM, optional
        A completed, converged FORM analysis, whose design point,
        transformation and settings SORM reuses. If ``None``, :meth:`run`
        first runs FORM with ``options.form``.
    on_failure : {"raise", "return"}, default "raise"
        On nonconvergence, raise :class:`~pystra.AnalysisError`, which carries
        the unconverged record in ``.result``, or return that record.

    Notes
    -----
    :meth:`run` returns a :class:`~pystra.results.SORMResult` with the
    curvatures and each formula's probability.
    """

    _options_type = SORMOptions

    def __init__(
        self, model, limit_state, *, options=None, form=None, on_failure="raise"
    ):
        if form is not None and not isinstance(form, FORM):
            raise TypeError("form must be a FORM analysis")
        self.form = form
        super().__init__(model, limit_state, options)
        self.on_failure = _check_on_failure(on_failure)
        if form is not None and self.options.form != FORMOptions():
            raise ModelError(
                "options.form applies only when SORM runs FORM; "
                "a supplied FORM analysis keeps its own settings"
            )
        self._reset_results()

    def _reset_results(self):
        """Invalidate previous SORM output before attempting another run."""
        self._results_valid = False
        self._betaHL = None
        self._kappa = None
        self._kappa_pf = None
        self._fit_type = None
        self._pf2_breitung = None
        self._betag_breitung = None
        self._pf2_breitung_m = None
        self._betag_breitung_m = None
        self._undefined = set()

    def _form_options(self):
        return (
            self.options.form
            if self._supplied_form is None
            else self._supplied_form.options
        )

    def _dependence(self):
        options = self._form_options()
        return options.transform, options.rosenblatt_order

    def _settings(self):
        return self._form_options()

    def _prepare_run(self, fit_type):
        """Run FORM if needed and require a converged design point."""
        self._reset_results()
        self._n_evaluations = 0
        if self._supplied_form is not None and self.options.form != FORMOptions():
            raise ModelError(
                "options.form applies only when SORM runs FORM; "
                "a supplied FORM analysis keeps its own settings"
            )
        if self._supplied_form is None:
            form = FORM(
                self.model,
                self.limit_state,
                options=self.options.form,
                on_failure="return",
            )
            self._form = form
            form.run()
        _check_form(self.form, self.model, self.limit_state)
        self.init_run()
        _check_coordinates(self.form, self.transform)
        self._fit_type = fit_type

    def run(self):
        """Run SORM with the fit in ``options.fit``.

        Curve fitting uses the eigenvalues of the Hessian at the design
        point; point fitting locates points on the failure surface by Newton
        iteration. FORM is run first if no FORM analysis was supplied.

        Returns
        -------
        SORMResult
            The immutable record of this run.

        Raises
        ------
        AnalysisError
            If FORM did not converge, a fitting point cannot be found, or the
            formula is undefined for the fitted curvatures, unless
            ``on_failure="return"``.
        """
        try:
            if self.options.fit == "curve":
                result = self._run_curvefit()
            else:
                result = self._run_pointfit()
        except AnalysisError as error:
            result = self._failed_result(str(error))
            if self.on_failure == "raise":
                raise AnalysisError(str(error), result) from error
            return result
        if not result.converged and self.on_failure == "raise":
            raise AnalysisError(result.message, result)
        return result

    def _failed_result(self, message):
        """Return the record of a fit that could not be completed."""
        return SORMResult(
            method="SORM",
            status="not_converged",
            message=message,
            n_limit_state_evaluations=self._n_evaluations,
            variable_names=tuple(self.model.get_variables()),
            failure_probability=None,
            beta=None,
            form=FORMResult.from_analysis(self.form),
            fit=self.options.fit,
            curvatures=None,
            formula=self.options.formula,
            options=self.options,
            approximations={"breitung": None, "modified_breitung": None},
        )

    def _run_curvefit(self):
        """Run curve fitting and return a :class:`SORMResult`; FORM must have converged."""
        self._prepare_run("cf")
        hess_G = self._compute_hessian()
        R1 = self._orthonormal_matrix()
        A = R1 @ hess_G @ R1.T / np.linalg.norm(self.form._gradient)
        kappa, _ = np.linalg.eig(A[:-1, :-1])
        kappa = np.real_if_close(kappa, tol=1e7)
        self._betaHL = self.form._beta
        self._kappa = np.sort(kappa)
        self._pf_breitung(self._betaHL, self._kappa)
        self._pf_breitung_m(self._betaHL, self._kappa)
        self._results_valid = True
        return self._result()

    def _run_pointfit(self):
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
        beta = self.form._beta
        nrv = self.form._alpha.shape[1]
        R1 = self._orthonormal_matrix()
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
        self._betaHL = self.form._beta
        self._kappa_pf = np.vstack([kappa_minus, kappa_plus])
        self._kappa = np.sort(0.5 * (kappa_minus + kappa_plus))

        # Compute failure probabilities
        self._pf_breitung_pf(beta, kappa_minus, kappa_plus)
        self._pf_breitung_m_pf(beta, kappa_minus, kappa_plus)

        self._results_valid = True
        return self._result()

    def _result(self):
        """Return the immutable record of the completed fit."""
        approximations = {
            "breitung": self._pf2_breitung,
            "modified_breitung": self._pf2_breitung_m,
        }
        for name in self._undefined:
            approximations[name] = None
        estimate = approximations[self.options.formula]
        # The index comes from the log probability, so it stays finite when
        # the probability itself underflows to zero.
        index = {
            "breitung": self._betag_breitung,
            "modified_breitung": self._betag_breitung_m,
        }[self.options.formula]
        return SORMResult(
            method="SORM",
            status="converged" if estimate is not None else "not_converged",
            message=(
                "Fitted"
                if estimate is not None
                else f"The {self.options.formula.replace('_', ' ')} formula is undefined for the fitted curvatures"
            ),
            n_limit_state_evaluations=self._n_evaluations,
            variable_names=tuple(self.model.get_variables()),
            failure_probability=estimate,
            beta=float(index) if estimate is not None else None,
            form=FORMResult.from_analysis(self.form),
            fit="curve" if self._fit_type == "cf" else "point",
            curvatures=self._kappa if self._fit_type == "cf" else self._kappa_pf,
            formula=self.options.formula,
            options=self.options,
            approximations=approximations,
        )

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
            G, grad = self._evaluate_lsf(x_col, calc_gradient=True)
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
            log_pf = log_ndtr(-beta) + np.sum(
                np.logaddexp(-0.5 * np.log(terms_plus), -0.5 * np.log(terms_minus))
                - np.log(2.0)
            )
            self._pf2_breitung = float(np.exp(log_pf))
            self._betag_breitung = float(-ndtri_exp(log_pf))
        else:
            self._undefined.add("breitung")
            self._pf2_breitung = 0.0
            self._betag_breitung = 0.0

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
        psi = np.exp(normal.logpdf(beta) - log_ndtr(-beta))
        terms_plus = 1 + psi * kappa_plus
        terms_minus = 1 + psi * kappa_minus
        is_invalid = np.any(terms_plus <= 0) or np.any(terms_minus <= 0)

        if not is_invalid:
            log_pf = log_ndtr(-beta) + np.sum(
                np.logaddexp(-0.5 * np.log(terms_plus), -0.5 * np.log(terms_minus))
                - np.log(2.0)
            )
            self._pf2_breitung_m = float(np.exp(log_pf))
            self._betag_breitung_m = float(-ndtri_exp(log_pf))
        else:
            self._undefined.add("modified_breitung")
            self._pf2_breitung_m = 0.0
            self._betag_breitung_m = 0.0

    def _pf_breitung(self, beta, kappa):
        """
        Calculates the probability of failure and generalized reliability
        index using [Breitung1984]_ formula. This formula is good for higher
        values of beta.
        """
        is_invalid = np.any(kappa < -1 / beta)
        if not is_invalid:
            log_pf = log_ndtr(-beta) - 0.5 * np.sum(np.log1p(beta * kappa))
            self._pf2_breitung = float(np.exp(log_pf))
            self._betag_breitung = float(-ndtri_exp(log_pf))
        else:
            self._undefined.add("breitung")
            self._pf2_breitung = 0.0
            self._betag_breitung = 0.0

    def _pf_breitung_m(self, beta, kappa):
        """
        Calculates the probability of failure and generalized reliability
        index using Brietung's formula ([Breitung1984]_) as modified by Hohenbichler and
        Rackwitz [Hohenbichler1988]_. This formula is better for lower values of beta.

        """
        k = np.exp(normal.logpdf(beta) - log_ndtr(-beta))
        is_invalid = np.any(kappa < -1 / k)
        if not is_invalid:
            log_pf = log_ndtr(-beta) - 0.5 * np.sum(np.log1p(k * kappa))
            self._pf2_breitung_m = float(np.exp(log_pf))
            self._betag_breitung_m = float(-ndtri_exp(log_pf))
        else:
            self._undefined.add("modified_breitung")
            self._pf2_breitung_m = 0.0
            self._betag_breitung_m = 0.0

    def _compute_hessian(self, diff_type=None):
        """
        Computes the matrix of second derivatives using forward finite
        difference, using the evaluation of the gradient already done
        for FORM, at the design point

        Could use numdifftools as external library instead

        """

        h = 1 / self.options.ffd_parameter
        nrv = self.form._alpha.shape[1]
        hess_G = np.zeros((nrv, nrv))

        if diff_type is None:
            # Differentiation based on the gradients
            x0 = self.form._design_point_x()
            _, grad_g0 = self._evaluate_lsf(x0[:, np.newaxis], calc_gradient=True)
            u0 = self.form._u
            for i in range(nrv):
                u1 = np.copy(u0)
                u1[i] += h
                x1 = self.transform.u_to_x(u1, self.model.get_marginal_distributions())
                _, grad_g1 = self._evaluate_lsf(x1[:, np.newaxis], calc_gradient=True)
                hess_G[:, i] = ((grad_g1 - grad_g0) / h).reshape(nrv)

        else:
            # FERUM-implementation using a mix of central and foward diffs
            # It would be good if u_to_x could take an nvr x nx matrix and
            # return the corresponding x-matrix. This would make it easier to
            # add more numerical differentiation schemes.
            u0 = self.form._u

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
            all_G, _ = self._evaluate_lsf(all_x, calc_gradient=False)
            all_G = all_G.squeeze()
            all_G_plus = all_G[:nrv]
            all_G_minus = all_G[nrv : 2 * nrv]
            all_G_both = all_G[2 * nrv : :]
            G = self.form._G

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

    def _evaluate_lsf(self, x, calc_gradient=False, u_space=True):
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
            G, grad = self._lsf(x0, gradient=True)
            grad = np.transpose(grad)
            if u_space:
                marg = self.model.get_marginal_distributions()
                u = self.transform.x_to_u(x0, marg)
                J_u_x = self.transform.jacobian(u, x0, marg)
                J_x_u = np.linalg.inv(J_u_x)
                grad = np.dot(grad, J_x_u)
        else:
            G, _ = self._lsf(x0, gradient=True)

        return G, grad

    def _orthonormal_matrix(self):
        """
        Computes the rotation matrix of the standard normal coordinate
        space where the design point is located at Beta along the last
        axis.
        """
        alpha = self.form._alpha.ravel()
        nrv = len(alpha)
        # Preserve the established tangent axes away from degeneracy. If
        # alpha is orthogonal to the last coordinate, omit its largest
        # component's axis instead, so the seed vectors remain independent.
        pivot = nrv - 1
        if abs(alpha[pivot]) <= np.sqrt(np.finfo(float).eps):
            pivot = int(np.argmax(np.abs(alpha)))
        axes = np.eye(nrv)
        A = np.column_stack(
            [alpha, *[axes[i] for i in reversed(range(nrv)) if i != pivot]]
        )
        Q = self._gram_schmidt(A)
        # And undo this rotation with a final 90 dgree rotation as order of
        # the column vector entries is not relevant
        R1 = np.rot90(Q)
        return R1

    def _gram_schmidt(self, A):
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
