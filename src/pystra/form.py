#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import warnings
from scipy.stats import norm as normal
from .analysis import AnalysisObject
from .results import FORMResult
from .correlation import set_modified_correlation_matrix

__all__ = ["FORM"]


class FORM(AnalysisObject):
    r"""Find a FORM design point and approximate the failure probability.

    Parameters
    ----------
    stochastic_model : StochasticModel
        Named physical variables and their joint probability model.
    limit_state : LimitState
        Physical response, with negative values indicating failure.
    analysis_options : AnalysisOptions, optional
        Transformation, derivative, iteration and convergence settings.

    Notes
    -----
    Call :meth:`run` to obtain an immutable :class:`~pystra.results.FORMResult`.
    Check its ``converged`` flag before using its probability or design point.
    Numerical convergence does not establish the accuracy of the local
    boundary approximation or exclude competing failure regions.

    Independent standard-normal coordinates are the usual choice. Explicit
    spherical Student-t generalized Nataf instead uses a Student-t half-space
    tail. ``get_beta()`` is the signed geometric distance;
    ``get_equivalent_beta()`` and ``FORMResult.beta`` are normal-equivalent.

    See :doc:`/guides/form_sorm` for usage and
    :doc:`/theory/design_point_methods` for the formulation and references.
    """

    supports_spherical_space = True

    def __init__(self, stochastic_model=None, limit_state=None, analysis_options=None):
        super().__init__(
            stochastic_model=stochastic_model,
            limit_state=limit_state,
            analysis_options=analysis_options,
        )

        self.i = None
        self.u = None
        self.x = None
        self.J = None
        self.G = None
        self.Go = None
        self.gradient = None
        self.alpha = None
        self.gamma = None
        self.d = None
        self.step = None
        self.beta = None
        self.Pf = None
        self.converged = False
        self.e1 = None
        self.e2 = None

    def run(self) -> FORMResult:
        """
        Execute FORM and return an immutable :class:`FORMResult` snapshot.
        """
        self.results_valid = False
        self.converged = False
        self.beta = self.Pf = None
        self.i = self.e1 = self.e2 = None
        imax = self.options.get_imax()
        if (
            isinstance(imax, bool)
            or not isinstance(imax, (int, np.integer))
            or imax < 1
        ):
            raise ValueError("FORM iteration limit must be a positive integer")

        self.init_run()

        # Compute starting point for the algorithm
        self.compute_starting_point()

        # Iterations
        # Set parameters for the iterative loop
        # Initialize counter
        i = 1
        # Convergence is achieved when convergence is set to True
        convergence = False

        # loope
        while not convergence:
            if self.options.get_print_output():
                print(".......................................")
                print("Now carrying out iteration number:", i)

            # Compute Transformation from u to x space
            self.compute_transformation()

            # Compute the Jacobian
            self.compute_jacobian()

            # Evaluate limit-state function and its gradient
            self.compute_limit_state()
            if (
                not np.all(np.isfinite(self.G))
                or not np.all(np.isfinite(self.gradient))
                or not np.isfinite(np.linalg.norm(self.gradient))
                or np.linalg.norm(self.gradient) == 0
            ):
                raise ValueError(
                    "FORM requires finite limit-state values and a nonzero finite gradient"
                )

            # Set scale parameter Go and inform about struct. resp.
            if i == 1:
                self.Go = self.G
                if self.options.get_print_output():
                    print("Value of limit-state function in the first step:", self.G)

            # Compute alpha vector
            self.compute_alpha()

            # Compute gamma vector
            self.compute_gamma()

            # Check convergence
            scale = float(np.abs(self.Go).item()) or 1.0
            e1 = float(np.abs(self.G).item()) / scale
            e2 = np.linalg.norm(self.u - self.alpha.dot(self.u).dot(self.alpha))
            self.e1, self.e2 = e1, float(e2)
            condition1 = e1 < self.options.get_e1()
            condition2 = e2 < self.options.get_e2()
            condition3 = i == self.options.get_imax()
            if self.options.get_print_output():
                print(f"e1 = {e1:1.6e} , e2 = {e2:1.6e}")

            if condition1 and condition2 or condition3:
                self.i = i
                self.converged = bool(condition1 and condition2)
                convergence = True

            # space for some recording stuff

            # Take a step if convergence is not achieved
            if not convergence:
                # Determine search direction
                self.compute_search_direction()

                # Determine step size
                self.get_step_size()

                # Determine new trial point
                u_new = self.u + self.step * self.d

                # Prepare for a new round in the loop
                self.u = u_new[0]  # np.transpose(u_new)
                i += 1

        # Compute beta value
        self.compute_beta()

        # Compute failure probability
        self.compute_failure_probability()
        self.results_valid = self.converged
        if not self.converged:
            warnings.warn(
                "FORM did not converge within the iteration limit", RuntimeWarning
            )

        # Show Results
        if self.options.get_print_output() and self.results_valid:
            self.show_results()
        return FORMResult.from_analysis(self)

    def compute_starting_point(self):
        """Compute starting point for the algorithm"""
        x = np.array([])
        marg = self.model.get_marginal_distributions()
        for i in range(len(marg)):
            x = np.append(x, marg[i].get_start_point())
        self.u = self.transform.x_to_u(x, marg)

    def compute_transformation(self):
        """Compute transformation from u to x space"""
        self.x = np.transpose(
            [self.transform.u_to_x(self.u, self.model.get_marginal_distributions())]
        )

    def compute_jacobian(self):
        """Compute the Jacobian"""
        J_u_x = self.transform.jacobian(
            self.u, self.x, self.model.get_marginal_distributions()
        )
        J_x_u = np.linalg.inv(J_u_x)
        self.J = J_x_u

    def compute_limit_state(self):
        """Evaluate limit-state function and its gradient"""
        G, gradient = self.limitstate.evaluate_lsf(self.x, self.model, self.options)
        self.G = G
        self.gradient = np.dot(np.transpose(gradient), self.J)

    def compute_alpha(self):
        """Compute alpha vector"""
        self.alpha = -self.gradient * np.linalg.norm(self.gradient) ** (-1)

    def compute_gamma(self):
        """Compute gamma vector"""
        self.gamma = np.diag(np.sqrt(np.diag(np.dot(self.J, np.transpose(self.J)))))
        # Importance vector gamma
        # matmult = np.dot(np.dot(self.alpha, self.J), self.gamma)
        # importance_vector_gamma = matmult / np.linalg.norm(matmult)

    def compute_search_direction(self):
        """Determine search direction"""
        self.d = (
            self.G * np.linalg.norm(self.gradient) ** (-1) + self.alpha.dot(self.u)
        ) * self.alpha - self.u

    def get_step_size(self):
        """Determine step size"""
        if self.options.get_step_size() == 0:
            self.step = self.compute_step_size(
                self.G,
                self.gradient,
                self.u,
                self.d,
            )
        else:
            self.step = self.options.get_step_size()

    def compute_step_size(self, G, gradient, u, d):
        """Calculate the step size for the calculation

        :Returns:
            - step_size (float): Returns the value of the step size.
        """
        c = (np.linalg.norm(u) * np.linalg.norm(gradient) ** (-1)) * 2 + 10
        merit = 0.5 * (np.linalg.norm(u)) ** 2 + c * np.absolute(G)

        ntrial = 6
        """
        .. note::

             TODO: change fix value to a variable
        """

        Trial_step_size = np.array([0.5 ** np.arange(0, ntrial)])

        uT = np.reshape([u], (len(u), -1))
        dT = np.transpose(d)  # np.reshape(d,(len(d),-1))
        # zero = np.array([np.ones(ntrial)])
        # zeroT = np.reshape(zero, (len(zero), -1))
        Trial_u = np.dot(uT, np.array([np.ones(ntrial)])) + np.dot(dT, Trial_step_size)
        Trial_x = np.zeros(Trial_u.shape)
        for j in range(ntrial):
            trial_x = self.transform.u_to_x(
                Trial_u[:, j], self.model.get_marginal_distributions()
            )
            Trial_x[:, j] = np.transpose(trial_x)

        if self.options.get_multi_proc() == 0:
            print("Error: function not yet implemented")
        if self.options.get_multi_proc() == 1:
            Trial_G, _ = self.limitstate.evaluate_lsf(
                Trial_x, self.model, self.options, "no"
            )
            Merit_new = np.zeros(ntrial)

            for j in range(ntrial):
                merit_new = 0.5 * (
                    np.linalg.norm(Trial_u[:, j])
                ) ** 2 + c * np.absolute(Trial_G[0][j])
                Merit_new[j] = merit_new

            trial_step_size = Trial_step_size[0][0]
            merit_new = Merit_new[0]

            j = 0

            while merit_new > merit and j < ntrial:
                trial_step_size = Trial_step_size[0][j]
                merit_new = Merit_new[j]
                j += 1
                if j == ntrial and merit_new > merit:
                    if self.options.get_print_output():
                        print(
                            "The step size has been reduced by a factor of 1/",
                            2**ntrial,
                        )
        step_size = trial_step_size
        return step_size

    def compute_beta(self):
        """Compute beta value"""
        self.beta = np.dot(self.alpha, self.u)[0]

    def compute_failure_probability(self):
        """Compute probability of failure"""
        marginal = getattr(self.transform, "standard_marginal", normal)
        self.Pf = float(marginal.sf(self.beta))

    def get_equivalent_beta(self):
        """Return -Phi^-1(Pf), including for non-normal standard spaces."""
        if not self.results_valid:
            raise ValueError("Analysis has no valid result")
        return float(-normal.ppf(self.Pf))

    def show_results(self):
        """Show results"""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        n_hyphen = self.N_HYPH
        print("")
        print("=" * n_hyphen)
        print("")
        print(" RESULTS FROM RUNNING FORM RELIABILITY ANALYSIS")
        print("")
        print(" Number of iterations:     ", self.i)
        print(" Reliability index beta:   ", self.beta)
        print(" Failure probability:      ", self.Pf)
        print(
            " Number of calls to the limit-state function:",
            self.get_no_function_calls(),
        )
        print("")
        print("=" * n_hyphen)
        print("")

    def show_detailed_output(self):
        """Get detailed output to console"""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        names = self.model.get_variables().keys()
        consts = self.model.get_constants()
        u_star = self.get_design_point()
        x_star = self.get_design_point(uspace=False)
        alpha = self.get_alpha()

        n_hyphen = self.N_HYPH
        print("")
        print("=" * n_hyphen)
        print("FORM")
        print("=" * n_hyphen)
        print("{:15s} \t {:1.10e}".format("Pf", self.Pf))
        print("{:15s} \t {:2.10f}".format("BetaHL", self.beta))
        print(
            "{:15s} \t {:d}".format("Model Evaluations", self.model.get_call_function())
        )
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

    def get_beta(self):
        """Returns the beta value

        :Returns:
          - beta (float): Returns the beta value
        """
        return self.beta

    def get_failure(self):
        """Returns the probability of failure

        :Returns:
          - Pf (float): Returns the probability of failure
        """
        return self.Pf

    def get_design_point(self, uspace=True):
        """Returns the design point, defaults to u-space

        :Returns:
          - u (float): Returns the design point in u- or x-space
        """
        if uspace:
            return self.u
        else:
            return self.transform.u_to_x(
                self.u, self.model.get_marginal_distributions()
            )

    def get_alpha(self, as_dict=False):
        """Returns the alpha vector

        :Returns:
          - alpha (np.array): Returns the alpha vector
        """
        if as_dict:
            names = self.model.get_names()
            alphas = self.alpha[0]
            alpha_dict = {name: alpha for alpha, name in zip(alphas, names)}
            return alpha_dict
        return self.alpha[0]

    def get_no_function_calls(self):
        """
        Returns the number of function evaluations used

        :Returns:
          - n (int): Returns the number of function evaluations

        """
        return self.model.get_call_function()
