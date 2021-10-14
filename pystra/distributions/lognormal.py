#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import lognorm
from .distribution import Distribution


class Lognormal(Distribution):
    """Lognormal distribution

    :Arguments:
      - name (str):         Name of the random variable
      - mean (float):       Mean or lamb
      - stdv (float):       Standard deviation or zeta\n
      - input_type (any):   Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n

    Note: Could use scipy to do the heavy lifting. However, there is a small
    performance hit, so for this common dist use bespoke implementation
    for the PDF, CDF.
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):

        self.dist_type = "Lognormal"

        if input_type is None:
            cov = stdv / mean
            zeta = (np.log(1 + cov ** 2)) ** 0.5
            lamb = np.log(mean) - 0.5 * zeta ** 2
        else:
            lamb = mean
            zeta = stdv

        self.lamb = lamb
        self.zeta = zeta

        # Could use scipy to do the heavy lifting. However, there is a small
        # performance hit, so for this common dist use bespoke implementation
        # for the PDF, CDF.
        # Careful: the scipy parametrization is tricky!
        self.dist_obj = lognorm(scale=np.exp(lamb), s=zeta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
        )

    # Overriding base class implementations for speed

    def pdf(self, x):
        """
        Probability density function
        Note: asssumes x>0 for performance, scipy manages this appropriately
        """
        z = (np.log(x) - self.lamb) / self.zeta
        p = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * self.zeta * x)
        return p  # self.lognormal.pdf(x)

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        z = (np.log(x) - self.lamb) / self.zeta
        p = 0.5 + math.erf(z / np.sqrt(2)) / 2
        return p  # self.lognormal.cdf(x)

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        x = np.exp(u * self.zeta + self.lamb)
        return x

    def x_to_u(self, x):
        """
        Transformation from x to u
        Note: asssumes x>0 for performance
        """
        u = (np.log(x) - self.lamb) / self.zeta
        return u
