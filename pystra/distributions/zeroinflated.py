# -*- coding: utf-8 -*-

import numpy as np
from .distribution import Distribution


class ZeroInflated(Distribution):
    """
    A Zero-Inflated rendering of the provided distribution.

    Variable loads sometimes have values of zero when they are not occurring.
    This distribution creates a mixed distribution where there is a certain
    probability `p` of a zero value, otherwise with a probability `1-p` a
    realization of the provided distribution occurs.

    :Attributes:
      - name (str):             Name of the random variable\n
      - mean (float):           Mean\n
      - stdv (float):           Standard deviation\n
      - dist (Distribution):    Distribution to zero-inflate
      - p (float):              Probability of zero
      - input_type (any):       Change meaning of mean and stdv\n
      - startpoint (float):     Start point for seach\n
    """

    def __init__(self, name, dist, p, input_type=None, startpoint=None):

        if not isinstance(dist, Distribution):
            raise Exception(
                f"ZeroInflated distribution requires input of type {type(Distribution)}"
            )
        if p < 0.0:
            raise Exception("ZeroInflated probability must be nonnegative")
        if p >= 1.0:
            raise Exception("ZeroInflated probability must be < 1.0")

        self.dist = dist
        self.p = p
        self.q = 1 - self.p
        self.zero_tol = 1e-6
        m, s = self._get_stats()

        super().__init__(
            name=name,
            mean=m,
            stdv=s,
            startpoint=startpoint,
        )

        self.dist_type = "ZeroInflated"

    def pdf(self, x):
        """
        Probability density function
        """
        x = np.atleast_1d(x)
        zipdf = self.dist.pdf(x) * self.q
        indx = (x > -self.zero_tol) & (x < self.zero_tol)
        zipdf[indx] += self.p
        return zipdf

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        x = np.atleast_1d(x)
        zicdf = self.dist.cdf(x) * self.q
        indx = x > -self.zero_tol
        zicdf[indx] += self.p
        return zicdf

    def ppf(self, p):
        """
        inverse cumulative distribution function
        """
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

        Refs:
        https://stats.stackexchange.com/questions/18661/mean-and-variance-of-a-zero-inflated-poisson-distribution
        https://stats.stackexchange.com/questions/310022/expected-value-of-the-square-of-a-random-variable
        """

        mean = self.q * self.dist.mean
        stdv = np.sqrt(
            self.q * self.dist.stdv**2 + self.p * self.q * self.dist.mean**2
        )

        return mean, stdv

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
        self._update_stats()

    def _update_stats(self):
        """
        Updates the mean and stdv estimates - used for sensitivity analysis
        where the parent distribution params may change after instantiation
        """
        m, s = self._get_stats()
        self.mean = m
        self.stdv = s
