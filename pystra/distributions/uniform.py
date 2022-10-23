#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import uniform

from .distribution import Distribution


class Uniform(Distribution):
    """Uniform distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean or a\n
      - stdv (float): Standard deviation or b\n
      - input_type (any): Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):

        if input_type is None:
            a = mean - 3**0.5 * stdv
            b = mean + 3**0.5 * stdv
        else:
            a = mean
            b = stdv

        self.a = a
        self.b = b

        # use scipy to do the heavy lifting
        self.dist_obj = uniform(loc=a, scale=b - a)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "Uniform"

    # Overriding these for performance

    def u_to_x(self, u):
        """
        Transformation from u to x

        Note: serious performance hit if scipy normal.cdf used here
        """
        x = self.a + (self.b - self.a) * self.std_normal.cdf(u)
        return x

    def x_to_u(self, x):
        """
        Transformation from x to u
        """
        u = self.std_normal.ppf(self.cdf(x))
        return u
