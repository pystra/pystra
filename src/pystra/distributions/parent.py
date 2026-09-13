#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as sp
import scipy.optimize as opt

from .distribution import Distribution


class MaxParent(Distribution):
    """Parent distribution of the provided distribution which represents
    the distribution of maxima of a random variable.

    For example, given an annual maximum distribution of imposed load, find
    the parent distribution of imposed load, if the load is applied 6 times
    per year.

    :Attributes:
      - name (str):             Name of the random variable\n
      - mean (float):           Mean\n
      - stdv (float):           Standard deviation\n
      - maximum (Distribution): Distribution of maximum object
      - N (float):              Power to which distribution is raised
      - input_type (any):       Change meaning of mean and stdv\n
      - startpoint (float):     Start point for seach\n
    """

    def __init__(self, name, max_dist, N, input_type=None, startpoint=None):
        if not isinstance(max_dist, Distribution):
            raise Exception(
                f"MaxParent distribution of maximum requires input of type {type(Distribution)}"
            )
        if N < 1.0:
            raise Exception("MaxParent exponent must be >= 1.0")

        self.max_dist = max_dist
        self.N = N
        m, s = self._get_stats()

        super().__init__(
            name=name,
            mean=m,
            stdv=s,
            startpoint=startpoint,
        )

        self.dist_type = "MaxParent"

    def pdf(self, x):
        """
        Probability density function
        """
        pdf = self.max_dist.pdf(x)
        cdf = self.cdf(x)
        p = pdf / (self.N * cdf ** (self.N - 1))
        return p

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        P = (self.max_dist.cdf(x)) ** (1 / self.N)
        return P

    def ppf(self, p):
        """
        inverse cumulative distribution function
        """
        scalar_input = np.isscalar(p)
        p = np.atleast_1d(np.asarray(p, dtype=float))
        x = np.empty_like(p, dtype=float)

        log_tiny = np.log(np.finfo(float).tiny)
        log_one = np.log(np.nextafter(1.0, 0.0))
        center = self.max_dist.mean
        if not np.isfinite(center):
            center = 0.0
        step0 = self.max_dist.stdv
        if not np.isfinite(step0) or step0 <= 0:
            step0 = 1.0

        if (
            type(self.max_dist).cdf is Distribution.cdf
            and self.max_dist.dist_obj is not None
        ):
            logcdf = self.max_dist.dist_obj.logcdf
        elif self.max_dist.dist_type == "Normal":
            logcdf = lambda q: sp.log_ndtr(
                (q - self.max_dist.mean) / self.max_dist.stdv
            )
        else:

            def logcdf(q):
                with np.errstate(divide="ignore"):
                    return np.log(self.max_dist.cdf(q))

        for index, p_val in np.ndenumerate(p):
            if p_val <= 0:
                x[index] = self.max_dist.ppf(0)
                continue
            if p_val >= 1:
                x[index] = self.max_dist.ppf(1)
                continue

            target_log_cdf = self.N * np.log(p_val)
            if log_tiny <= target_log_cdf <= log_one:
                x[index] = self.max_dist.ppf(np.exp(target_log_cdf))
                continue

            residual = lambda q: logcdf(q) - target_log_cdf
            step = step0
            lower = center - step
            upper = center + step
            for _ in range(100):
                lower_res = residual(lower)
                upper_res = residual(upper)
                if lower_res <= 0 <= upper_res:
                    x[index] = opt.brentq(
                        residual, lower, upper, xtol=1e-12, rtol=1e-12
                    )
                    break
                if lower_res > 0:
                    upper = lower
                    lower -= step
                if upper_res < 0:
                    lower = upper
                    upper += step
                step *= 2
            else:
                raise RuntimeError("Could not bracket MaxParent inverse CDF.")

        if scalar_input:
            return x.item()
        return x

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        p = self.std_normal.cdf(u)
        x = self.ppf(p)
        return x

    def x_to_u(self, x):
        """
        Transformation from x to u
        """
        u = self.std_normal.ppf(self.cdf(x))
        return u

    def jacobian(self, u, x):
        """
        Compute the Jacobian (e.g. Lemaire, eq. 4.9)
        """
        pdf1 = self.pdf(x)
        pdf2 = self.std_normal.pdf(u)
        J = np.diag(pdf1 / pdf2)
        return J

    def _get_stats(self):
        """
        Since the closed form expression of mean and stdv for the distribution of the
        parent from a maximum distribution is complex, and since we really only need
        them for default starting points, just estimate through simulation.
        """
        p = np.random.random(100)
        x = self.ppf(p)
        mean = x.mean()
        stdv = x.std()

        return mean, stdv

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
        Updates the mean and stdv estimates - used for sensitivity analysis
        where the parent distribution params may change after instantiation
        """
        m, s = self._get_stats()
        self.mean = m
        self.stdv = s
