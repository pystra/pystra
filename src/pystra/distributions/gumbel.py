#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import gumbel_l, gumbel_r as gumbel

from .distribution import Distribution

__all__ = ["Gumbel", "GumbelMin"]


class Gumbel(Distribution):
    """Gumbel distribution for maxima: the Type I extreme value distribution.

    :Attributes:
        - name (str):     Name of the random variable\n
        - mean (float): Mean or mu\n
        - std (float): Standard deviation or beta\n
        - input_type (any): Change meaning of mean and std\n
        - start_point (float): Start point for seach\n
    """

    def __init__(self, name, mean, std, input_type=None, start_point=None):
        if input_type is None:
            mu = mean - 0.5772156649 * std * np.sqrt(6) / np.pi
            scale = std * np.sqrt(6) / np.pi
        else:
            mu = mean
            scale = std

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
      - mean (float): Mean or mu\n
      - std (float): Standard deviation or beta\n
      - input_type (any): Change meaning of mean and std\n
      - start_point (float): Start point for seach\n
    """

    def __init__(self, name, mean, std, input_type=None, start_point=None):
        if input_type is None:
            beta = np.pi / (std * np.sqrt(6))
            mu = mean + (0.5772156649 * std * np.sqrt(6)) / np.pi
        else:
            mu = mean
            beta = std

        # use scipy to do the heavy lifting
        self.dist_obj = gumbel_l(loc=mu, scale=1 / beta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "GumbelMin"
