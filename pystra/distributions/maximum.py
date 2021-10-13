#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

from .distribution import Distribution


class Maximum(Distribution):
    """Distribution of maximima from the passed in parent distribution

    :Attributes:
      - name (str):             Name of the random variable\n
      - mean (float):           Mean\n
      - stdv (float):           Standard deviation\n
      - parent (Distribution):  Parent distribution object
      - N (float):              Power to which distribution is raised
      - input_type (any):       Change meaning of mean and stdv\n
      - startpoint (float):     Start point for seach\n
    """

    def __init__(self, name, parent, N, input_type=None, startpoint=None):

        self.dist_type = "Maximum"

        if not isinstance(parent, Distribution):
            raise Exception(
                f"Maximum parent requires input of type {type(Distribution)}"
            )
        if N < 1.0:
            raise Exception("Maximum exponent must be >= 1.0")

        self.parent = parent
        self.N = N
        m, s = self._get_stats()

        super().__init__(
            name=name, mean=m, stdv=s, startpoint=startpoint,
        )

    def pdf(self, x):
        """
        Probability density function
        """
        pdf = self.parent.pdf(x)
        cdf = 1.0
        if self.N > 1.0:
            cdf = self.parent.cdf(x)
        p = self.N * pdf * cdf ** (self.N - 1)
        return p

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        P = (self.parent.cdf(x)) ** self.N
        return P

    def inv_cdf(self, p):
        """
        inverse cumulative distribution function
        """
        p = np.atleast_1d(p)
        x = np.zeros_like(p)
        x0 = self.parent.mean
        for i, p_val in enumerate(p):
            par = opt.fmin(self.zero_distn, x0, args=(p_val,), disp=False)
            x[i] = par[0]
        return x

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        p = self.std_normal.cdf(u)
        x = self.inv_cdf(p)
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
        maxima from a parent distribution is complex, and since we really only need
        them for default starting points, just estimate through simulation.
        """
        p = np.random.random(100)
        x = self.inv_cdf(p)
        mean = x.mean()
        stdv = x.std()

        return mean, stdv

    def zero_distn(self, x, *args):
        p = args
        cdf = self.cdf(x)
        zero = np.absolute(cdf - p)
        return zero
