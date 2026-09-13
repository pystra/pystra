"""First-order reliability analysis using a design-point search."""

import warnings

import numpy as np
from scipy.stats import norm as normal

from .analysis import AnalysisObject, _check_on_failure
from ._form_reuse import _problem_state
from ..errors import AnalysisError
from ..options import FORMOptions
from ..results import FORMResult

__all__ = ["FORM"]


class FORM(AnalysisObject):
    r"""Find a FORM design point and approximate the failure probability.

    Parameters
    ----------
    model : StochasticModel
        Named physical variables and their joint probability model.
    limit_state : LimitState
        Physical response, with negative values indicating failure.
    options : FORMOptions, optional
        Iteration, convergence, differentiation and transformation settings.
    on_failure : {"raise", "return"}, default "raise"
        On nonconvergence, raise :class:`~pystra.AnalysisError`, which carries
        the unconverged record in ``.result``, or return that record with a warning.

    Notes
    -----
    Call :meth:`run` to obtain an immutable :class:`~pystra.results.FORMResult`.
    Check its ``converged`` flag before using its probability or design point.
    Numerical convergence does not establish the accuracy of the local
    boundary approximation or exclude competing failure regions.

    Independent standard normal coordinates are the usual choice. Explicit
    spherical Student-t generalized Nataf instead uses a Student-t half-space
    tail. ``FORMResult.design_index`` is the signed geometric distance and
    ``FORMResult.beta`` the normal-equivalent index.

    See :doc:`/guides/form_sorm` for usage and
    :doc:`/theory/design_point_methods` for the formulation and references.
    """

    supports_spherical_space = True
    _options_type = FORMOptions

    def __init__(self, model, limit_state, *, options=None, on_failure="raise"):
        super().__init__(model, limit_state, options)
        self.on_failure = _check_on_failure(on_failure)

        self._i = None
        self._u = None
        self._x = None
        self._J = None
        self._G = None
        self._Go = None
        self._gradient = None
        self._alpha = None
        self._gamma = None
        self._d = None
        self._step = None
        self._beta = None
        self._Pf = None
        self._converged = False
        self._e1 = None
        self._e2 = None
        self._last_result = None

    def run(self) -> FORMResult:
        """
        Execute FORM and return an immutable :class:`FORMResult` snapshot.
        """
        self._last_result = None
        self._results_valid = False
        self._converged = False
        self._beta = self._Pf = None
        self._i = self._e1 = self._e2 = None
        self.init_run()

        # Compute starting point for the algorithm
        self._compute_starting_point()

        # Iterations
        # Set parameters for the iterative loop
        # Initialize counter
        i = 1
        # Convergence is achieved when convergence is set to True
        convergence = False

        # loope
        while not convergence:
            # Compute Transformation from u to x space
            self._compute_transformation()

            # Compute the Jacobian
            self._compute_jacobian()

            # Evaluate limit-state function and its gradient
            self._compute_limit_state()
            if (
                not np.all(np.isfinite(self._G))
                or not np.all(np.isfinite(self._gradient))
                or not np.isfinite(np.linalg.norm(self._gradient))
                or np.linalg.norm(self._gradient) == 0
            ):
                raise ValueError(
                    "FORM requires finite limit-state values and a nonzero finite gradient"
                )

            # Set scale parameter Go and inform about struct. resp.
            if i == 1:
                self._Go = self._G
            # Compute alpha vector
            self._compute_alpha()

            # Compute gamma vector
            self._compute_gamma()

            # Check convergence
            scale = float(np.abs(self._Go).item()) or 1.0
            e1 = float(np.abs(self._G).item()) / scale
            e2 = np.linalg.norm(self._u - self._alpha.dot(self._u).dot(self._alpha))
            self._e1, self._e2 = e1, float(e2)
            condition1 = e1 < self.options.limit_state_tolerance
            condition2 = e2 < self.options.gradient_tolerance
            condition3 = i == self.options.max_iterations
            if condition1 and condition2 or condition3:
                self._i = i
                self._converged = bool(condition1 and condition2)
                convergence = True

            # space for some recording stuff

            # Take a step if convergence is not achieved
            if not convergence:
                # Determine search direction
                self._compute_search_direction()

                # Determine step size
                self._get_step_size()

                # Determine new trial point
                u_new = self._u + self._step * self._d

                # Prepare for a new round in the loop
                self._u = u_new[0]  # np.transpose(u_new)
                i += 1

        # Compute beta value
        self._compute_beta()

        # Compute failure probability
        self._compute_failure_probability()
        self._results_valid = self._converged
        self._run_expression = self.limit_state.expression
        self._run_options = self.options
        self._run_model_state = _problem_state(self.model)
        result = self._last_result = FORMResult.from_analysis(self)
        if not self._converged:
            message = "FORM did not converge within the iteration limit"
            if self.on_failure == "raise":
                raise AnalysisError(message, result)
            warnings.warn(message, RuntimeWarning)
        return result

    def _compute_starting_point(self):
        """Compute starting point for the algorithm"""
        x = np.array([])
        marg = self.model.get_marginal_distributions()
        for i in range(len(marg)):
            x = np.append(x, marg[i].start_point)
        self._u = self.transform.x_to_u(x, marg)

    def _compute_transformation(self):
        """Compute transformation from u to x space"""
        self._x = np.transpose(
            [self.transform.u_to_x(self._u, self.model.get_marginal_distributions())]
        )

    def _compute_jacobian(self):
        """Compute the Jacobian"""
        J_u_x = self.transform.jacobian_u_wrt_x(
            self._u, self._x, self.model.get_marginal_distributions()
        )
        J_x_u = np.linalg.inv(J_u_x)
        self._J = J_x_u

    def _compute_limit_state(self):
        """Evaluate limit-state function and its gradient"""
        G, gradient = self._lsf(self._x, gradient=True)
        self._G = G
        self._gradient = np.dot(np.transpose(gradient), self._J)

    def _compute_alpha(self):
        """Compute alpha vector"""
        self._alpha = -self._gradient * np.linalg.norm(self._gradient) ** (-1)

    def _compute_gamma(self):
        """Compute gamma vector"""
        self._gamma = np.diag(np.sqrt(np.diag(np.dot(self._J, np.transpose(self._J)))))
        # Importance vector gamma
        # matmult = np.dot(np.dot(self.alpha, self.J), self.gamma)
        # importance_vector_gamma = matmult / np.linalg.norm(matmult)

    def _compute_search_direction(self):
        """Determine search direction"""
        self._d = (
            self._G * np.linalg.norm(self._gradient) ** (-1) + self._alpha.dot(self._u)
        ) * self._alpha - self._u

    def _get_step_size(self):
        """Determine step size"""
        if self.options.step_size == 0:
            self._step = self._compute_step_size(
                self._G,
                self._gradient,
                self._u,
                self._d,
            )
        else:
            self._step = self.options.step_size

    def _compute_step_size(self, G, gradient, u, d):
        """Choose a step by comparing trial points with the current merit.

        Parameters
        ----------
        G : ndarray
            Current limit-state value, shape (1, 1).
        gradient : ndarray
            Gradient in standard space, shape (1, n_variables).
        u : ndarray
            Current point in standard space, shape (n_variables,).
        d : ndarray
            Search direction, shape (1, n_variables).

        Returns
        -------
        float
            Selected step from the six trials 1, 1/2, ..., 1/32. The last
            trial is returned if none improves the merit.
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

        Trial_G, _ = self._lsf(Trial_x)
        Merit_new = np.zeros(ntrial)

        for j in range(ntrial):
            merit_new = 0.5 * (np.linalg.norm(Trial_u[:, j])) ** 2 + c * np.absolute(
                Trial_G[0][j]
            )
            Merit_new[j] = merit_new

        trial_step_size = Trial_step_size[0][0]
        merit_new = Merit_new[0]

        j = 0

        while merit_new > merit and j < ntrial:
            trial_step_size = Trial_step_size[0][j]
            merit_new = Merit_new[j]
            j += 1
        step_size = trial_step_size
        return step_size

    def _compute_beta(self):
        """Compute beta value"""
        self._beta = np.dot(self._alpha, self._u)[0]

    def _compute_failure_probability(self):
        """Compute probability of failure"""
        marginal = getattr(self.transform, "standard_marginal", normal)
        self._Pf = float(marginal.sf(self._beta))

    def _get_equivalent_beta(self):
        """Return -Phi^-1(Pf), including for non-normal standard spaces."""
        if not self._results_valid:
            raise ValueError("Analysis has no valid result")
        return float(-normal.ppf(self._Pf))

    def _design_point_x(self):
        """Return the design point in physical coordinates."""
        return self.transform.u_to_x(self._u, self.model.get_marginal_distributions())
