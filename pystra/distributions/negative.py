#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

from .distribution import Distribution


class Negative(Distribution):
    """Returns a distribution with negative support from the supplied
    distribution. That is, this 'flips' the distribution about zero.

    :Attributes:
      - name (str):             Name of the random variable\n
      - positive_dist (Distribution):  Distribution object to be flipped
      - input_type (any):       Change meaning of mean and stdv\n
      - startpoint (float):     Start point for seach\n
    """

    def __init__(self, name, positive_dist, input_type=None, startpoint=None):

        if not isinstance(positive_dist, Distribution):
            raise Exception(
                f"Negative positive_dist requires input of type {type(Distribution)}"
            )

        self.positive_dist = positive_dist

        super().__init__(
            name=name,
            mean=-positive_dist.mean,
            stdv=positive_dist.stdv,
            startpoint=startpoint,
        )

        self.dist_type = "Negative"

    def pdf(self, x):
        """
        Probability density function
        """
        pdf = self.positive_dist.pdf(-x)
        return pdf

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        P = self.positive_dist.cdf(-x)
        return P

    def inv_cdf(self, p):
        """
        inverse cumulative distribution function
        """
        x = self.positive_dist.inv_cdf(p)
        return -1 * x

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        p = self.std_normal.cdf(u)
        x = self.inv_cdf(p)
        return -1 * x

    def x_to_u(self, x):
        """
        Transformation from x to u
        """
        u = self.std_normal.ppf(self.cdf(-x))
        return u

    def jacobian(self, u, x):
        """
        Compute the Jacobian (e.g. Lemaire, eq. 4.9)
        """
        pdf1 = self.pdf(-x)
        pdf2 = self.std_normal.pdf(u)
        J = np.diag(pdf1 / pdf2)
        return J

    def set_location(self, loc=0):
        """
        Updating the positive_dist distribution location parameter.
        """
        self.positive_dist.set_location(-loc)
        self.update_stats()

    def set_scale(self, scale=1):
        """
        Updating the positive_dist distribution scale parameter.
        """
        self.positive_dist.set_scale(scale)
        self.update_stats()

    def zero_distn(self, x, *args):
        p = args
        cdf = self.cdf(-x)
        zero = np.absolute(cdf - p)
        return zero
