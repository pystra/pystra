#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

from .form import Form
from .analysis import AnalysisObject
from scipy.stats import norm as normal


class Sorm(AnalysisObject):
    r"""
    Second Order Reliability Method (SORM)

    Unlike FORM, this method approximates the failure surface in standard
    normal space using a quadratic surface. The basic approximation is given
    by [Breitung1984]_:

        .. math::

              p_f \\approx p_{f2} = \Phi(-\\beta) \\Pi_{i=1}^{n-1}\\[ 1 + \\kappa_i \\beta \\]^{-0.5}

    The corresponding generalized reliability index
    :math:`\\beta_G \\= -\\Phi^{-1}\\( 1-p_{f2}\\)` is reported. The Breitung
    approximation is asymptotic and accurate for higher values of \\beta.
    Also reported is the Hohenbichler and Rackwitz modification to Brietung's
    formula, which is more accurate for lower values of \\beta.

    :Attributes:
      - stochastic_model (StochasticModel): Information about the model
      - limit_state (LimitState): Information about the limit state
      - analysis_option (AnalysisOption): Option for the structural analysis
      - form (Form): Form object, if a FORM analysis has already been completed

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
            self.form = Form(
                stochastic_model=self.model,
                limit_state=self.limitstate,
                analysis_options=self.options,
            )
            self.form.run()
        else:
            self.form = form

        self.betaHL = 0
        self.kappa = 0
        self.pf2_breitung = 0
        self.betag_breitung = 0
        self.pf2_breitung_m = 0
        self.betag_breitung_m = 0
        # Could add Tvedt too

    def run(self, fit_type="cf"):
        """
        Run SORM analysis using either:
            Curve-fitting: fit_type == 'cf'
            Point-fitting: fit_type == 'pf'
        """
        self.results_valid = True
        self.init_run()

        if fit_type != "cf":
            raise ValueError("Point-Fitting not yet supported")
        else:
            self.run_curvefit()

    def run_curvefit(self):
        """
        Run SORM analysis using curve fitting
        """
        hess_G = self.computeHessian()
        R1 = self.orthonormal_matrix()
        A = R1 @ hess_G @ R1.T / np.linalg.norm(self.form.gradient)
        kappa, _ = np.linalg.eig(A[:-1, :-1])
        kappa = np.real_if_close(kappa, tol=1e7)
        self.betaHL = self.form.beta
        self.kappa = np.sort(kappa)
        self.pf_breitung(self.betaHL, self.kappa)
        self.pf_breitung_m(self.betaHL, self.kappa)
        if self.options.getPrintOutput():
            self.showResults()

    def pf_breitung(self, beta, kappa):
        """
        Calculates the probability of failure and generalized reliability
        index using [Breitung1984]_ formula. This formula is good for higher
        values of beta.
        """
        is_invalid = np.any(kappa < -1 / beta)
        if not is_invalid:
            self.pf2_breitung = normal.cdf(-beta) * np.prod(
                (1 + beta * kappa) ** (-0.5)
            )
            self.betag_breitung = -normal.ppf(self.pf2_breitung)
        else:
            print("*** SORM Breitung error, excessive curavtures")
            self.pf2_breitung = 0
            self.betag_breitung = 0

    def pf_breitung_m(self, beta, kappa):
        """
        Calculates the probability of failure and generalized reliability
        index using Brietung's formula ([Breitung1984]_) as modified by Hohenbichler and
        Rackwitz [Hohenbichler1988]_. This formula is better for lower values of beta.

        """
        k = normal.pdf(beta) / normal.cdf(-beta)
        is_invalid = np.any(kappa < -1 / k)
        if not is_invalid:
            self.pf2_breitung_m = normal.cdf(-beta) * np.prod((1 + k * kappa) ** (-0.5))
            self.betag_breitung_m = -normal.ppf(self.pf2_breitung_m)
        else:
            print("*** SORM Breitung Modified error")
            self.pf2_breitung_m = 0
            self.betag_breitung_m = 0

    def showResults(self):
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        n_hyphen = 54
        print("")
        print("=" * n_hyphen)
        print("")
        print("RESULTS FROM RUNNING SECOND ORDER RELIABILITY METHOD")
        print("")
        print("Generalized reliability index: ", self.betag_breitung[0])
        print("Probability of failure:        ", self.pf2_breitung[0])
        print("")
        for i, k in enumerate(self.kappa):
            print(f"Curvature {i+1}: {k}")
        print("=" * n_hyphen)
        print("")

    def showDetailedOutput(self):
        """Get detailed output to console"""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        names = self.model.getVariables().keys()
        consts = self.model.getConstants()
        u_star = self.form.getDesignPoint()
        x_star = self.form.getDesignPoint(uspace=False)
        alpha = self.form.getAlpha()
        betaHL = self.form.beta[0]
        pfFORM = self.form.Pf

        n_hyphen = 54
        print("")
        print("=" * n_hyphen)
        print("FORM/SORM")
        print("=" * n_hyphen)
        print("{:15s} \t\t {:1.10e}".format("Pf FORM", pfFORM[0]))
        print("{:15s} \t\t {:1.10e}".format("Pf SORM Breitung", self.pf2_breitung[0]))
        print(
            "{:15s} \t {:1.10e}".format("Pf SORM Breitung HR", self.pf2_breitung_m[0])
        )
        print("{:15s} \t\t {:2.10f}".format("Beta_HL", betaHL))
        print("{:15s} \t\t {:2.10f}".format("Beta_G Breitung", self.betag_breitung[0]))
        print(
            "{:15s} \t\t {:2.10f}".format(
                "Beta_G Breitung HR", self.betag_breitung_m[0]
            )
        )
        print(
            "{:15s} \t\t {:d}".format("Model Evaluations", self.model.getCallFunction())
        )

        print("-" * n_hyphen)
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

    def computeHessian(self, diff_type=None):
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
            x0 = self.form.getDesignPoint(uspace=False)
            _, grad_g0 = self.evaluateLSF(x0[:, np.newaxis], calc_gradient=True)
            u0 = self.form.getDesignPoint()
            for i in range(nrv):
                u1 = np.copy(u0)
                u1[i] += h
                x1 = self.transform.u_to_x(u1, self.model.getMarginalDistributions())
                _, grad_g1 = self.evaluateLSF(x1[:, np.newaxis], calc_gradient=True)
                hess_G[:, i] = ((grad_g1 - grad_g0) / h).reshape(nrv)

        else:
            # FERUM-implementation using a mix of central and foward diffs
            # It would be good if u_to_x could take an nvr x nx matrix and
            # return the corresponding x-matrix. This would make it easier to
            # add more numerical differentiation schemes.
            u0 = self.form.getDesignPoint()

            all_x_plus = np.zeros((nrv, nrv))
            all_x_minus = np.zeros((nrv, nrv))
            all_x_both = np.zeros((nrv, int(nrv * (nrv - 1) / 2)))

            marg = self.model.getMarginalDistributions()
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
                        u_both, self.model.getMarginalDistributions()
                    )
                    all_x_both[:, int((i - 1) * (i) / 2) + j] = x_both

            # Assemble all x-space vecs, solve for G, then separate
            all_x = np.concatenate((all_x_plus, all_x_minus, all_x_both), axis=1)
            all_G, _ = self.evaluateLSF(all_x, calc_gradient=False)
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

    def evaluateLSF(self, x, calc_gradient=False, u_space=True):
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
                marg = self.model.getMarginalDistributions()
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
        A = np.eye(nrv)
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

        A = np.asfarray(A)
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
