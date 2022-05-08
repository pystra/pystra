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

        if input_type is None:
            # infer parameters from the moments
            self._update_params(mean, stdv)
        else:
            # parameters directly passed in
            self.lamb = mean
            self.zeta = stdv

        # Could use scipy to do the heavy lifting. However, there is a small
        # performance hit, so for this common dist use bespoke implementation
        # for the PDF, CDF.
        # Careful: the scipy parametrization is tricky!
        self.dist_obj = lognorm(scale=np.exp(self.lamb), s=self.zeta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "Lognormal"

    def _update_params(self, mean, stdv):
        cov = stdv / mean
        self.zeta = (np.log(1 + cov**2)) ** 0.5
        self.lamb = np.log(mean) - 0.5 * self.zeta**2

    # Overriding base class implementations for speed

    def pdf(self, x):
        """
        Probability density function
        Note: asssumes x>0 for performance, scipy manages this appropriately
        """
        z = (np.log(x) - self.lamb) / self.zeta
        p = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * self.zeta * x)
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

    def set_location(self, loc=0):
        """
        Updating the distribution location parameter.
        For Lognormal, even though we have a SciPy object, it's not being used in the
        functions above for performance, so we need to update pe.arams directly.
        """

        self._update_params(loc, self.stdv)
        self.mean = loc

    def set_scale(self, scale=1):
        """
        Updating the distribution scale parameter.
        For Lognormal, even though we have a SciPy object, it's not being used in the
        functions above for performance, so we need to update params directly.
        """
        self._update_params(self.mean, scale)
        self.stdv = scale
