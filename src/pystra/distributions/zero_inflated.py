"""Zero inflated marginal distribution."""

import numpy as np
from scipy import special as sp
from .distribution import Distribution, _piecewise
from ..errors import ModelError

__all__ = ["ZeroInflated"]


class ZeroInflated(Distribution):
    """Mixture of a point mass at zero and a supplied distribution.

    With probability p the value is zero; with probability 1 - p a value
    is drawn from dist.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    dist : Distribution
        Distribution realized when the variable is not set to zero.
    p : float
        Probability of setting the variable to zero; must satisfy 0 <= p < 1.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ()

    def _parameter_values(self):
        return {"dist": self.dist, "p": self.p}

    @property
    def sensitivity_params(self):
        """No generic moment perturbation; replace the constructor inputs."""
        return {}

    def __init__(self, name, dist, p, *, start_point=None):
        if not isinstance(dist, Distribution):
            raise ModelError(
                f"ZeroInflated distribution requires input of type {type(Distribution)}"
            )
        if p < 0.0:
            raise ModelError("ZeroInflated probability must be nonnegative")
        if p >= 1.0:
            raise ModelError("ZeroInflated probability must be < 1.0")

        self.dist = dist
        self.p = p
        self.q = 1 - self.p
        self.zero_tol = 1e-6
        m, s = self._get_stats()

        super().__init__(
            name=name,
            mean=m,
            std=s,
            start_point=start_point,
        )

        self.dist_type = "ZeroInflated"

    def pdf(self, x):
        """
        Probability density function
        """
        scalar_input = np.isscalar(x)
        x = np.atleast_1d(x)
        zipdf = self.dist.pdf(x) * self.q
        indx = (x > -self.zero_tol) & (x < self.zero_tol)
        zipdf[indx] += self.p
        if scalar_input:
            return zipdf.item()
        return zipdf

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        scalar_input = np.isscalar(x)
        x = np.atleast_1d(x)
        zicdf = self.dist.cdf(x) * self.q
        indx = x > -self.zero_tol
        zicdf[indx] += self.p
        if scalar_input:
            return zicdf.item()
        return zicdf

    def ppf(self, p):
        """
        inverse cumulative distribution function
        """
        scalar_input = np.isscalar(p)
        p = np.atleast_1d(p)
        x = np.zeros_like(p)

        # Probability of a value less than zero
        p0 = self.dist.cdf(0.0)
        qp0 = self.q * p0
        qp0p = qp0 + self.p
        # values below zero
        indx0 = p < qp0
        x[indx0] = self.dist.ppf(p[indx0] / self.q)
        # values at zero
        indxp = (p > qp0) & (p < qp0p)
        x[indxp] = 0.0
        # values above zero
        indx = p >= qp0p
        x[indx] = self.dist.ppf((p[indx] - self.p) / self.q)
        if scalar_input:
            return x.item()
        return x

    def sf(self, x):
        """Survival function; the zero atom lies on the CDF side."""
        x = np.asarray(x, dtype=float)
        above = x > -self.zero_tol
        return np.where(above, self.q * self.dist.sf(x), 1 - self.q * self.dist.cdf(x))[
            ()
        ]

    def isf(self, q):
        """Inverse survival function, from the parent's upper tail."""
        q = np.asarray(q, dtype=float)
        upper = q < self.q * self.dist.sf(0.0)
        return _piecewise(
            q, upper, lambda v: self.ppf(1 - v), lambda v: self.dist.isf(v / self.q)
        )

    def _lower_logcdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(
                x > -self.zero_tol,
                np.log(self.cdf(x)),
                np.log(self.q) + self.dist.logcdf(x),
            )[()]

    def _upper_logsf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(
                x > -self.zero_tol,
                np.log(self.q) + self.dist.logsf(x),
                np.log(self.sf(x)),
            )[()]

    def _lower_quantile_log(self, logp):
        logp = np.asarray(logp, dtype=float)
        below = logp - np.log(self.q) < self.dist.logcdf(0.0)
        return _piecewise(
            logp,
            below,
            lambda v: self.ppf(np.exp(v)),
            lambda v: self.dist._ppf_log(v - np.log(self.q)),
        )

    def _upper_quantile_log(self, logq):
        logq = np.asarray(logq, dtype=float)
        above = logq - np.log(self.q) < self.dist.logsf(0.0)
        return _piecewise(
            logq,
            above,
            lambda v: self.ppf(-np.expm1(v)),
            lambda v: self.dist._isf_log(v - np.log(self.q)),
        )

    def _tail_quantiles(self, x, prob, z, quantile_log, log_tail, direction):
        # The zero atom makes the CDF non-invertible, so quantiles cannot be
        # checked against it. The log quantiles already send both tails
        # through the parent and keep points in the atom at exactly zero.
        index = np.flatnonzero(prob < 1e-8)
        if index.size:
            x[index] = quantile_log(sp.log_ndtr(z[index]))
        return x

    def _get_stats(self):
        """
        Since the closed form expression of mean and std for the distribution of the
        parent from a maximum distribution is complex, and since we really only need
        them for default starting points, just estimate through simulation.

        Refs:
        https://stats.stackexchange.com/questions/18661/mean-and-variance-of-a-zero-inflated-poisson-distribution
        https://stats.stackexchange.com/questions/310022/expected-value-of-the-square-of-a-random-variable
        """

        mean = self.q * self.dist.mean
        std = np.sqrt(self.q * self.dist.std**2 + self.p * self.q * self.dist.mean**2)

        return mean, std

    def set_location(self, loc=0):
        """
        Updating the zero-inflated distribution location parameter.
        """
        self.dist.set_location(loc)
        self._update_stats()

    def set_scale(self, scale=1):
        """
        Updating the zero-inflated distribution scale parameter.
        """
        self.dist.set_scale(scale)
        self._update_stats()

    def set_zero_probability(self, p):
        """
        Update the zero-inflated probability.
        """
        self.p = p
        self.q = 1 - self.p
        self._update_stats()

    def _update_stats(self):
        """
        Updates the mean and std estimates - used for sensitivity analysis
        where the parent distribution params may change after instantiation
        """
        m, s = self._get_stats()
        self._mean = m
        self._std = s
