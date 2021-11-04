#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import gumbel_l as gumbel

from .distribution import Distribution


class TypeIsmallestValue(Distribution):
    """Type I smallest value distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean or mu\n
      - stdv (float): Standard deviation or beta\n
      - input_type (any): Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):

        if input_type is None:
            beta = np.pi / (stdv * np.sqrt(6))
            mu = mean + (0.5772156649 * stdv * np.sqrt(6)) / np.pi
        else:
            mu = mean
            beta = stdv

        # use scipy to do the heavy lifting
        self.dist_obj = gumbel(loc=mu, scale=1 / beta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "TypeIsmallestValue"
