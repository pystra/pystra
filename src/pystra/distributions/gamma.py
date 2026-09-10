#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import gamma

from .distribution import Distribution

__all__ = ["Gamma"]


class Gamma(Distribution):
    """Gamma distribution

    :Attributes:
      - name (str):         Name of the random variable\n
      - mean (float):       Mean or beta\n
      - std (float):       Standard deviation or k\n
      - input_type (any):   Change meaning of mean and std\n
      - start_point (float): Start point for seach\n
    """

    def __init__(self, name, mean, std, input_type=None, start_point=None):
        if input_type is None:
            beta = mean / (std**2)
            alpha = mean**2 / (std**2)
        else:
            beta = mean
            alpha = std

        # use scipy to do the heavy lifting
        self.dist_obj = gamma(a=alpha, scale=1 / beta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Gamma"
