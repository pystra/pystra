#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import gumbel_l, gumbel_r as gumbel

from .distribution import Distribution, _uses_native_parameters

__all__ = ["Gumbel", "GumbelMin"]


class Gumbel(Distribution):
    """Gumbel distribution for maxima: the Type I extreme value distribution.

    :Attributes:
        - name (str):     Name of the random variable\n
        - mean (float): Mean\n
        - std (float): Standard deviation\n
        - loc (float): Location, given instead of mean and std\n
        - scale (float): Scale, given instead of mean and std\n
        - start_point (float): Start point for seach\n
    """

    def __init__(
        self, name, mean=None, std=None, *, loc=None, scale=None, start_point=None
    ):
        if _uses_native_parameters(self, mean, std, loc=loc, scale=scale):
            mu = loc
        else:
            mu = mean - 0.5772156649 * std * np.sqrt(6) / np.pi
            scale = std * np.sqrt(6) / np.pi

        # use scipy to do the heavy lifting
        self.dist_obj = gumbel(loc=mu, scale=scale)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Gumbel"


class GumbelMin(Distribution):
    """Gumbel distribution for minima: the Type I smallest value distribution.

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean\n
      - std (float): Standard deviation\n
      - loc (float): Location, given instead of mean and std\n
      - scale (float): Scale, given instead of mean and std\n
      - start_point (float): Start point for seach\n
    """

    def __init__(
        self, name, mean=None, std=None, *, loc=None, scale=None, start_point=None
    ):
        if _uses_native_parameters(self, mean, std, loc=loc, scale=scale):
            mu = loc
        else:
            beta = np.pi / (std * np.sqrt(6))
            mu = mean + (0.5772156649 * std * np.sqrt(6)) / np.pi
            scale = 1 / beta

        # use scipy to do the heavy lifting
        self.dist_obj = gumbel_l(loc=mu, scale=scale)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "GumbelMin"
