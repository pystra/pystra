"""Monte Carlo reliability estimates and sampled response distributions."""

import numpy as np
from scipy.special import logsumexp, ndtri_exp

from .analysis import AnalysisObject, _check_rng, _generator
from ..distributions import StdNormal
from ..options import SimulationOptions
from ..results import DistributionAnalysisResult, SimulationResult

__all__ = [
    "MonteCarlo",
    "CrudeMonteCarlo",
    "DistributionAnalysis",
]


class MonteCarlo(AnalysisObject):
    """Base implementation for Monte Carlo simulation analyses.

    Random samples in physical or standard space estimate the failure
    probability from limit-state evaluations [Nowak2000]_ [Faber2009]_
    [Lemaire2010]_. Use CrudeMonteCarlo for direct sampling or a specialized
    subclass for a different sampling scheme.

    Parameters
    ----------
    model : StochasticModel
        Named physical variables and their joint probability model.
    limit_state : LimitState
        Physical response evaluated at the sample points.
    options : SimulationOptions, optional
        Sampling, convergence and transformation settings.
    rng : int, numpy.random.Generator or None, optional
        Random source; NumPy's global generator is not used. A seed recreates
        the same stream on every run, a generator advances its own state, and
        None draws fresh entropy.
    """

    _options_type = SimulationOptions
    _unused_options = ("bins",)

    def __init__(self, model, limit_state, *, options=None, rng=None):
        super().__init__(model, limit_state, options)
        self.options._require_defaults(type(self).__name__, self._unused_options)
        self.rng = _check_rng(rng)

        self._nrv = self.model.get_len_marginal_distributions()
        self.point = None
        self._covariance = None
        self._cholesky_covariance = None
        self._inverse_covariance = None
        self._sum_q = None
        self._log_sum_q = None
        self._log_sum_q2 = None
        self._q_bar = None
        self._log_q_bar = None
        self._cov_q_bar = None
        self._log_factors = None
        self._k = None
        self._done = None
        self._block_size = None
        self._u = None
        self._u_all = None
        self._x = None
        self._x_all = None
        self._beta = None
        self._Pf = None
        self._G = None
        self._all_G1 = None
        self._I = None
        self._q = None
        self._Pf = None
        self._beta = None
        self._approxMC_beta = None
        self._approxMC_beta_all = None
        self._all_X = None
        self._all_G = None
        self._bins = None

    def _set_point(self, point=None):
        """Set design point"""
        if point is None:
            self.point = np.zeros((self._nrv, 1))
        else:
            self.point = point

    def _compute_random_numbers(self):
        """Compute random numbers"""
        self._u = np.dot(self.point, [np.ones(self._block_size)]) + np.dot(
            self._cholesky_covariance,
            self._random.standard_normal((self._nrv, self._block_size)),
        )

    def _compute_transformation(self):
        """Transform each sample column from standard to physical space.

        The stored arrays have shape (n_variables, block_size), with rows in
        model variable order. Each transformation receives one sample column.
        """
        self._x = np.zeros((self._nrv, self._block_size))

        for i in range(self._block_size):
            self._x[:, i] = self.transform.u_to_x(
                self._u[:, i], self.model.get_marginal_distributions()
            )

    def _compute_limit_state(self):
        """Evaluate limit-state function"""
        G, _ = self._lsf(self._x)
        self._G = G

    def _compute_results(self):
        """Collect result of sampling"""
        self._I = np.zeros(self._block_size)
        indx = np.where(self._G[0] < 0)
        self._I[indx] = 1

    def _compute_sum_update(self):
        """Update summation"""
        # The determinant factor belongs in the exponent too: std**n can
        # overflow even when the final density ratio is representable.
        active = self._I != 0
        points = self._u[:, active]
        delta = points - np.asarray(self.point).reshape(-1, 1)
        log_q = self._log_factors + 0.5 * (
            np.sum(delta * (self._inverse_covariance @ delta), axis=0)
            - np.sum(points * points, axis=0)
        )
        self._q = np.zeros(self._block_size)
        self._q[active] = np.exp(log_q)
        if log_q.size:
            # SciPy before 1.15 rejects an empty array in logsumexp
            self._log_sum_q = np.logaddexp(self._log_sum_q, logsumexp(log_q))
            self._log_sum_q2 = np.logaddexp(self._log_sum_q2, logsumexp(2 * log_q))
        self._sum_q += np.sum(self._q)

    def _compute_coefficient_of_variation(self):
        """Compute relative uncertainty without squaring tiny probabilities."""
        n = self._k - 1
        if np.isfinite(self._log_sum_q):
            self._log_q_bar[n] = self._log_sum_q - np.log(self._k)
            self._q_bar[n] = self._sum_q / self._k
            log_ratio = np.log(self._k) + self._log_sum_q2 - 2 * self._log_sum_q
            self._cov_q_bar[n] = np.sqrt(max(0.0, np.expm1(log_ratio)) / self._k)
            if self._cov_q_bar[n] == 0:
                self._cov_q_bar[n] = 1.0
        else:
            self._q_bar[n] = 0
            self._cov_q_bar[n] = 1.0

    def _compute_percent_done(self):
        """Compute percent done"""
        if int(self._k * self.options.n_samples ** (-1) * 20) > self._done:
            self._done = int(self._k * self.options.n_samples ** (-1) * 20)

    def _compute_failure_probability(self):
        """Compute probability of failure"""
        if self._sum_q > 0:
            self._Pf = self._q_bar[self._k - 1]
        else:
            self._Pf = 0

    def _compute_beta(self):
        """Convert the probability estimate to a normal-equivalent index.

        No observed failures gives an infinite estimated index, not zero.
        This describes the point estimate; a finite sample cannot establish
        that the true failure probability is zero.
        """
        self._beta = float(
            -StdNormal.ppf(self._Pf)
            if self._Pf > 0
            else -ndtri_exp(self._log_sum_q - np.log(self._k))
        )

    def _compute_bins(self, samples):
        """Return an optimal amount of bins for a histogram

        :Returns:
          - bins (int): Returns amount on bins
        """

        if self.options.bins is not None:
            bins = self.options.bins
        else:
            bins = np.ceil(4 * np.sqrt(np.sqrt(samples)))
        return bins


class CrudeMonteCarlo(MonteCarlo):
    """Crude Monte Carlo simulation (CMC)

    The Crude Monte Carlo simulation (CMC) is the most simple form and
    corresponds to a direct application of Equation (24). A large number
    :math:`n` of samples are simulated for the set of random variables
    :math:`{\\bf X}`. All samples that lead to a failure are counted :math:`n_f`
    and after all simulations the probability of failure :math:`p_f` may be
    estimated by [Faber2009]_

    .. math::

               \\tilde{p}_f = \\frac{n_f}{n}

    Theoretically, an infinite number of simulations will provide an exact
    probability of failure. However, time and the power of computers are
    limited; therefore, a suitable amount of simulations :math:`n` are required
    to achieve an acceptable level of accuracy.

    Parameters
    ----------
    model : StochasticModel
    limit_state : LimitState
    options : SimulationOptions, optional
    rng : int, numpy.random.Generator or None, optional
        Random source; NumPy's global generator is not used. A seed recreates
        the same stream on every run, a generator advances its own state, and
        None draws fresh entropy.
    point : ndarray, optional
        Center of the sampling density in standard coordinates; the origin
        by default.
    """

    def __init__(self, model, limit_state, *, options=None, point=None, rng=None):
        super().__init__(model, limit_state, options=options, rng=rng)
        self.point = point

    def run(self):
        """Run the simulation and return a :class:`SimulationResult`."""
        self._results_valid = False
        self._Pf = self._beta = None

        self.init_run()
        self._random = _generator(self.rng)

        # Set point for crude Monte Carlo / importance sampling
        self._set_point(self.point)
        self._ends = []

        # Initialize variables
        self._initialize_variables()

        self._k = 0
        while self._k < self.options.n_samples:
            self._block_size = min(
                self.options.block_size, self.options.n_samples - self._k
            )
            self._k += self._block_size
            self._ends.append(self._k)
            # Computation of the random numbers
            self._compute_random_numbers()

            # Compute transformation from u to x space
            self._compute_transformation()

            # Evaluate limit-state function
            self._compute_limit_state()

            # Collect result of sampling: if g < 0 , I = 1 , else I = 0
            self._compute_results()

            # Update sums
            self._compute_sum_update()

            # Compute coefficient of variation (of pf)
            self._compute_coefficient_of_variation()

            # Coumpute percent done
            self._compute_percent_done()

            # stroing all values of the limit-state function
            if self._u_all is None:
                self._all_G1 = self._G
            else:
                self._all_G1 = np.append(self._all_G1, self._G)

            # storing all input values in the Gaussian space
            if self._u_all is None:
                self._u_all = self._u
            else:
                self._u_all = np.append(self._u_all, self._u)

            # storing all input values in the physical space
            if self._x_all is None:
                self._x_all = self._x
            else:
                self._x_all = np.append(self._x_all, self._x)

            # compute approximative beta
            self._approxMC_beta = np.sqrt(
                np.array(
                    [np.sum(self._u[:, i] ** 2) for i in range(self._u[0, :].__len__())]
                )
            )
            # storing approximative beta
            if self._approxMC_beta_all is None:
                self._approxMC_beta_all = self._approxMC_beta
            else:
                self._approxMC_beta_all = np.append(
                    self._approxMC_beta_all, self._approxMC_beta
                )

            # Check convergence
            if self._cov_q_bar[self._k - 1] <= self.options.target_cov:
                break

        # Compute failure probability
        self._compute_failure_probability()

        # Compute beta value
        self._compute_beta()

        # Show Results
        result = self._result()
        self._results_valid = True
        return result

    def _diagnostics(self):
        """Return the convergence history for the result's diagnostics."""
        ends = np.array(self._ends)
        probability = self._q_bar[ends - 1]
        cov = np.where(
            np.isfinite(self._log_q_bar[ends - 1]), self._cov_q_bar[ends - 1], np.inf
        )
        return {
            "history": {
                "n_samples": ends,
                "failure_probability": probability,
                "coefficient_of_variation": cov,
            }
        }

    def _result(self):
        """Return the immutable record of the completed run."""
        pf = float(self._Pf)
        cov = (
            float(self._cov_q_bar[self._k - 1])
            if np.isfinite(self._log_sum_q)
            else np.inf
        )
        target = self.options.target_cov
        met = target == 0 or cov <= target
        return SimulationResult(
            method=type(self).__name__,
            status="completed" if met else "precision_not_met",
            message=(
                "Requested sample budget completed"
                if target == 0
                else (
                    f"Reached the target coefficient of variation, {target}"
                    if met
                    else f"Sample budget used before the target coefficient of variation, {target}"
                )
            ),
            n_limit_state_evaluations=self._n_evaluations,
            variable_names=tuple(self.model.get_variables()),
            failure_probability=pf,
            beta=float(self._beta),
            coefficient_of_variation=cov,
            n_samples=int(self._k),
            diagnostics=self._diagnostics(),
            options=self.options,
        )

    def _initialize_variables(self):
        """Initialization of the simulation variables"""
        stdv = self.options.sampling_std
        samples = self.options.n_samples
        # Establish covariance matrix, its Cholesky decomposition, and its inverse
        self._covariance = stdv**2 * np.eye(self._nrv)
        self._cholesky_covariance = stdv * np.eye(self._nrv)
        self._inverse_covariance = 1 * (stdv**2) ** (-1) * np.eye(self._nrv)

        # Initializations
        self._sum_q = 0
        self._log_sum_q = -np.inf
        self._log_sum_q2 = -np.inf
        self._q_bar = np.zeros(samples)
        self._log_q_bar = np.full(samples, -np.inf)
        self._cov_q_bar = np.empty(samples)
        self._cov_q_bar[:] = np.nan

        # Pre-compute some factors to minimize computations inside simulation loop
        self._log_factors = self._nrv * np.log(stdv)
        self._cov_q_bar[0] = 1.0
        self._done = 0


class DistributionAnalysis(MonteCarlo):
    """Distribution Analysis

    To analyze the random variables, used in the limit-state function, a
    numerical distribution analysis based on Monte Carlo simulation can be
    performed.

    Parameters
    ----------
    model : StochasticModel
    limit_state : LimitState
    options : SimulationOptions, optional
    rng : int, numpy.random.Generator or None, optional
        Random source; NumPy's global generator is not used. A seed recreates
        the same stream on every run, a generator advances its own state, and
        None draws fresh entropy.
    """

    _unused_options = ("target_cov",)

    def __init__(self, model, limit_state, *, options=None, rng=None):
        super().__init__(model, limit_state, options=options, rng=rng)

    def run(self):
        """Sample the model and return a :class:`DistributionAnalysisResult`."""
        self._results_valid = True

        self.init_run()
        self._random = _generator(self.rng)

        # Set point for crude Monte Carlo / importance sampling
        self._set_point()

        # Initialize variables # Different
        self._initialize_variables()

        self._k = 0
        while self._k < self.options.n_samples:
            self._block_size = min(
                self.options.block_size, self.options.n_samples - self._k
            )
            self._k += self._block_size

            # Computation of the random numbers
            self._compute_random_numbers()

            # Comoute Transformation from u to x space
            self._compute_transformation()

            # Evaluate limit-state function and its gradient
            self._compute_limit_state()

            # Collect Data
            self._compute_data_update()

            # Coumpute percent done
            self._compute_percent_done()

        # Compute Distribution Data
        self._compute_distribution_data()

        # Show Results # Different
        return DistributionAnalysisResult(
            method="DistributionAnalysis",
            status="completed",
            message="Sampling completed",
            n_limit_state_evaluations=self._n_evaluations,
            variable_names=tuple(self.model.get_variables()),
            n_samples=int(self._k),
            bins=int(self._bins),
            samples_x=self._all_X.T,
            limit_state_values=np.ravel(self._all_G),
            options=self.options,
        )

    def _initialize_variables(self):
        """Initialization of the simulation variables"""
        stdv = self.options.sampling_std
        samples = self.options.n_samples
        # Establish covariance matrix, its Cholesky decomposition, and its inverse
        self._covariance = stdv**2 * np.eye(self._nrv)
        self._cholesky_covariance = stdv * np.eye(self._nrv)

        ng = 1

        self._all_X = np.zeros((self._nrv, samples))
        self._all_G = np.zeros((ng, samples))

        self._done = 0
        self._bins = self._compute_bins(samples)

    def _compute_data_update(self):
        """Update data"""
        indx = list(range((self._k - self._block_size), self._k))
        self._all_X[:, indx] = self._x
        self._all_G[:, indx] = self._G

    def _compute_distribution_data(self):
        """Compute data for the distributions"""
        x = self._all_G
        x = np.transpose(x)
        self._all_G = x
