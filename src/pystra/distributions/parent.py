#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .distribution import Distribution, _log1mexp, _piecewise
from ._moments import _quantile_moments
from ..errors import ModelError

__all__ = ["MaxParent"]


class MaxParent(Distribution):
    """Parent distribution of the provided distribution which represents
    the distribution of maxima of a random variable.

    For example, given an annual maximum distribution of imposed load, find
    the parent distribution of imposed load, if the load is applied 6 times
    per year.

    Moments are computed deterministically by Gauss-Hermite quadrature of the
    quantile function in standard normal space, with exact identities for
    Frechet and uniform parents. Successive rules are compared at relative
    tolerance 1e-7 and absolute tolerance 1e-9 in standardized units; a miss
    warns and keeps the finest estimate, which serves for start points and
    finite-difference steps. Nonfinite moments raise :class:`~pystra.ModelError`.

    :Attributes:
      - name (str):             Name of the random variable\n
      - mean (float):           Mean\n
      - std (float):           Standard deviation\n
      - maximum (Distribution): Distribution of maximum object
      - N (float):              Power to which distribution is raised
      - start_point (float):     Start point for seach\n
    """

    def __init__(self, name, max_dist, N, *, start_point=None):
        if not isinstance(max_dist, Distribution):
            raise ModelError(
                f"MaxParent distribution of maximum requires input of type {type(Distribution)}"
            )
        if not np.isfinite(N) or N < 1.0:
            raise ModelError("MaxParent exponent must be >= 1.0")

        self.max_dist = max_dist
        self.N = N
        m, s = self._get_stats()

        super().__init__(
            name=name,
            mean=m,
            std=s,
            start_point=start_point,
        )

        self.dist_type = "MaxParent"

    def pdf(self, x):
        """
        Probability density function, from log densities
        """
        with np.errstate(under="ignore"):
            return np.exp(self.logpdf(x))

    def cdf(self, x):
        """
        Cumulative distribution function, ``exp(log F_max(x) / N)``
        """
        with np.errstate(under="ignore"):
            return np.exp(self.logcdf(x))

    def ppf(self, p):
        """
        Inverse cumulative distribution function, from the maximum's tails
        """
        with np.errstate(divide="ignore"):
            return self._ppf_log(np.log(np.asarray(p, dtype=float)))

    def isf(self, q):
        """Inverse survival function."""
        with np.errstate(divide="ignore"):
            return self._isf_log(np.log(np.asarray(q, dtype=float)))

    def logpdf(self, x):
        """Log density."""
        logpdf = self.max_dist.logpdf(x) - np.log(self.N)
        if self.N == 1:
            return logpdf
        return logpdf + (1 / self.N - 1) * self.max_dist.logcdf(x)

    def logcdf(self, x):
        """Log CDF, the maximum's divided by ``N``."""
        return self.max_dist.logcdf(x) / self.N

    def sf(self, x):
        """Survival function ``1 - F_max(x)**(1/N)``."""
        return -np.expm1(self.logcdf(x))

    def logsf(self, x):
        """Log survival function."""
        # Once 1 - F**(1/N) is below 1e-200 it equals (1 - F) / N
        a = np.asarray(self.logcdf(x), dtype=float)
        return _piecewise(
            x,
            a > -1e-200,
            lambda v: _log1mexp(self.logcdf(v)),
            lambda v: self.max_dist.logsf(v) - np.log(self.N),
        )

    def _lower_quantile_log(self, logp):
        # F_max(x)**(1/N) = p, so the maximum's log CDF is N log(p)
        return self.max_dist._ppf_log(self.N * np.asarray(logp, dtype=float))

    def _upper_quantile_log(self, logq):
        # 1 - F_max(x)**(1/N) = q gives the maximum's survival
        # -expm1(N log1p(-q)), which is N q once N q is below 1e-200
        logq = np.asarray(logq, dtype=float)
        with np.errstate(divide="ignore", under="ignore"):
            q = np.exp(logq)
            log_max_sf = np.where(
                self.N * q < 1e-200,
                np.log(self.N) + logq,
                _log1mexp(self.N * np.log1p(-q)),
            )
        return self.max_dist._isf_log(log_max_sf)

    def _get_stats(self):
        """Compute moments deterministically with quantile integration."""
        if self.N == 1:
            return self.max_dist.mean, self.max_dist.std
        return _quantile_moments(self, self.max_dist.mean, self.max_dist.std)

    def set_location(self, loc=0):
        """
        Updating the parent distribution location parameter.
        """
        self.max_dist.set_location(loc)
        self._update_stats()

    def set_scale(self, scale=1):
        """
        Updating the parent distribution scale parameter.
        """
        self.max_dist.set_scale(scale)
        self._update_stats()

    def set_exponent(self, N=2):
        """
        Update the parent distribution exponent parameter.
        """
        self.N = N
        self._update_stats()

    def _update_stats(self):
        """
        Updates the mean and std estimates - used for sensitivity analysis
        where the parent distribution params may change after instantiation
        """
        m, s = self._get_stats()
        self._mean = m
        self._std = s
