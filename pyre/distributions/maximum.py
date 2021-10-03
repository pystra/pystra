#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

from .distribution import Distribution
from .normal import Normal


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
        if not isinstance(parent, Distribution):
            raise Exception(
                f"Maximum parent requires input of type {type(Distribution)}"
            )
        if N < 1.0:
            raise Exception("Maximum exponent must be >= 1.0")

        self.type = 18
        self.distribution = {18: "Maximum"}
        self.name = name
        self.parent = parent
        self.N = N
        self.mean, self.stdv = self._get_stats()
        self.input_type = None

        mean, stdv, p1, p2, p3, p4 = self.setMarginalDistribution()

        super().__init__(
            name, self.type, mean, stdv, startpoint, p1, p2, p3, p4, input_type
        )

    def setMarginalDistribution(self):
        """
        Compute the marginal distribution
        """
        return self.mean, self.stdv, self.parent, self.N, None, None

    @classmethod
    def pdf(self, x, parent=None, N=None, var_3=None, var_4=None):
        """
        Probability density function
        """
        pdf = parent.pdf(
            x, parent.getP1(), parent.getP2(), parent.getP3(), parent.getP4()
        )
        cdf = 1.0
        if N > 1.0:
            cdf = parent.cdf(
                x, parent.getP1(), parent.getP2(), parent.getP3(), parent.getP4()
            )
        p = N * pdf * cdf ** (N - 1)
        return p

    @classmethod
    def cdf(self, x, parent=None, N=None, var_3=None, var_4=None):
        """
        Cumulative distribution function
        """
        P = (
            parent.cdf(
                x, parent.getP1(), parent.getP2(), parent.getP3(), parent.getP4()
            )
        ) ** N
        return P

    def _inv_cdf(self, p):
        """
        inverse cumulative distribution function
        """
        x = np.zeros(len(p))
        x0 = self.parent.mean
        for i in range(len(p)):
            par = opt.fmin(zero_distn, x0, args=(self.parent, self.N, p[i]), disp=False)
            x[i] = par[0]
        return x

    @classmethod
    def u_to_x(self, u, marg, x=None):
        """
        Transformation from u to x
        """
        if x is None:
            x = np.zeros(len(u))
        for i in range(len(u)):
            parent = marg.getP1()
            n = marg.getP2()
            x0 = marg.getMean()
            p = Normal.cdf(u[i], 0, 1)
            par = opt.fmin(zero_distn, x0, args=(parent, n, p), disp=False)
            x[i] = par[0]
        return x

    @classmethod
    def x_to_u(self, x, marg, u=None):
        """
        Transformation from x to u
        """
        if u is None:
            u = np.zeros(len(x))
        for i in range(len(x)):
            u[i] = Normal.inv_cdf(Maximum.cdf(x[i], marg.getP1(), marg.getP2()))
        return u

    @classmethod
    def jacobian(self, u, x, marg, J=None):
        """
        Compute the Jacobian (e.g. Lemaire, eq. 4.9)
        """
        if J is None:
            J = np.zeros((len(marg), len(marg)))
        for i in range(len(marg)):
            pdf1 = Maximum.pdf(x[i], marg.getP1(), marg.getP2())
            pdf2 = Normal.pdf(u[i], 0, 1)
            J[i][i] = pdf1 * (pdf2) ** (-1)
        return J

    def _get_stats(self):
        """
        Since the closed form expression of mean and stdv for the distribution of the
        maxima from a parent distribution is complex, and since we really only need
        them for default starting points, just estimate through simulation.
        """
        p = np.random.random(100)
        x = self._inv_cdf(p)
        mean = x.mean()
        stdv = x.std()

        return mean, stdv


def zero_distn(x, *args):
    parent, n, p = args
    cdf = Maximum.cdf(x, parent, n)
    zero = np.absolute(cdf - p)
    return zero
