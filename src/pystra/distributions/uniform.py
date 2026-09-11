#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import uniform

from .distribution import Distribution, _uses_native_parameters

__all__ = ["Uniform"]


class Uniform(Distribution):
    """Uniform distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean\n
      - std (float): Standard deviation\n
      - lower (float): Lower bound, given instead of mean and std\n
      - upper (float): Upper bound, given instead of mean and std\n
      - start_point (float): Start point for seach\n
    """

    def __init__(
        self, name, mean=None, std=None, *, lower=None, upper=None, start_point=None
    ):
        if _uses_native_parameters(self, mean, std, lower=lower, upper=upper):
            a = lower
            b = upper
        else:
            a = mean - 3**0.5 * std
            b = mean + 3**0.5 * std

        self.a = a
        self.b = b

        # use scipy to do the heavy lifting
        self.dist_obj = uniform(loc=a, scale=b - a)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
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
