#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.optimize as opt

from .distribution import *
from .normal import *


class NormalN(Distribution):
    """Normal distribution raised to the power N
  
  :Attributes:
    - name (str):         Name of the random variable\n
    - mean (float):       Mean\n
    - stdv (float):       Standard deviation\n
    - N (float):          Power to which distribution is raised
    - input_type (any):   Change meaning of mean and stdv\n
    - startpoint (float): Start point for seach\n 
  """

    def __init__(self, name, mean, stdv, N, input_type=None, startpoint=None):
        self.type = 17
        self.distribution = {17: "NormalN"}
        self.mean = mean
        self.stdv = stdv
        self.N = N
        mean, stdv, p1, p2, p3, p4 = self.setMarginalDistribution()
        Distribution.__init__(
            self, name, self.type, mean, stdv, startpoint, p1, p2, p3, p4, input_type
        )

    def setMarginalDistribution(self):
        """Compute the marginal distribution  
    """
        return self.mean, self.stdv, self.mean, self.stdv, self.N, 0

    @classmethod
    def pdf(self, x, mean=None, stdv=None, N=None, var_4=None):
        """probability density function
    """
        cdf = Normal.cdf(x, mean, stdv)
        pdf = Normal.pdf(x, mean, stdv)
        p = N * pdf * cdf ** (N - 1)
        return p

    @classmethod
    def cdf(self, x, mean=None, stdv=None, N=None, var_4=None):
        """cumulative distribution function
    """
        P = (Normal.cdf(x, mean, stdv)) ** N
        return P

    @classmethod
    def u_to_x(self, u, marg, x=None):
        """Transformation from u to x
    """
        if x == None:
            x = np.zeros(len(u))
        for i in range(len(u)):
            mu = marg.getP1()
            sd = marg.getP2()
            n = marg.getP3()
            mean = marg.getMean()
            p = Normal.cdf(u[i], 0, 1)
            par = opt.fmin(zero_normaln, mean, args=(mu, sd, n, p), disp=False)
            x[i] = par[0]
        return x

    @classmethod
    def x_to_u(self, x, marg, u=None):
        """Transformation from x to u
    """
        if u == None:
            u = np.zeros(len(x))
        for i in range(len(x)):
            u[i] = Normal.inv_cdf(
                NormalN.cdf(x[i], marg.getP1(), marg.getP2(), marg.getP3())
            )
        return u

    @classmethod
    def jacobian(self, u, x, marg, J=None):
        """
        Compute the Jacobian (e.g. Lemaire, eq. 4.9)
        """
        if J == None:
            J = np.zeros((len(marg), len(marg)))
        for i in range(len(marg)):
            pdf1 = NormalN.pdf(x[i], marg.getP1(), marg.getP2(), marg.getP3())
            pdf2 = Normal.pdf(u[i], 0, 1)
            J[i][i] = pdf1 * (pdf2) ** (-1)
        return J


def zero_normaln(x, *args):
    m, s, n, p = args
    cdf = Normal.cdf(x, m, s)
    zero_norm = np.absolute(cdf ** n - p)
    return zero_norm
