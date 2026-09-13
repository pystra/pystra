#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
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

    _native_parameters = ("lower", "upper")

    def _parameter_values(self):
        return {"lower": self.a, "upper": self.b}

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
        Transformation from u to x, measured from the nearer bound
        """
        u = np.asarray(u, dtype=float)
        tail = self.std_normal.cdf(-np.abs(u))
        width = self.b - self.a
        if u.ndim == 0:
            return self.a + width * tail if u <= 0 else self.b - width * tail
        return np.where(u <= 0, self.a + width * tail, self.b - width * tail)[()]
