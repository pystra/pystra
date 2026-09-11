#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import gamma

from .distribution import Distribution, _uses_native_parameters

__all__ = ["Gamma"]


class Gamma(Distribution):
    """Gamma distribution

    :Attributes:
      - name (str):         Name of the random variable\n
      - mean (float):       Mean\n
      - std (float):       Standard deviation\n
      - rate (float): Rate, the inverse of the scale, given instead of mean and std\n
      - shape (float): Shape, given instead of mean and std\n
      - start_point (float): Start point for seach\n
    """

    def __init__(
        self, name, mean=None, std=None, *, rate=None, shape=None, start_point=None
    ):
        if _uses_native_parameters(self, mean, std, rate=rate, shape=shape):
            beta = rate
            alpha = shape
        else:
            beta = mean / (std**2)
            alpha = mean**2 / (std**2)

        # use scipy to do the heavy lifting
        self.dist_obj = gamma(a=alpha, scale=1 / beta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Gamma"
