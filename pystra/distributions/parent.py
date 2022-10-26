#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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
        p = np.atleast_1d(p)
        x = np.zeros_like(p)
        x0 = self.max_dist.mean
        for i, p_val in enumerate(p):
            par = opt.fmin(self.zero_distn, x0, args=(p_val,), disp=False)
            x[i] = par[0]
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

    def zero_distn(self, x, *args):
        p = args
        cdf = self.cdf(x)
        zero = np.absolute(cdf - p)
        return zero
