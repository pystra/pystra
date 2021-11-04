#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rayleigh

from .distribution import Distribution


class ShiftedRayleigh(Distribution):
    """Shifted Rayleigh distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean or a\n
      - stdv (float): Standard deviation or x_zero\n
      - input_type (any): Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):

        if input_type is None:
            a = stdv / ((2 - np.pi * 0.5) ** 0.5)
            x_zero = mean - stdv * (np.pi / (4 - np.pi)) ** 0.5
        else:
            a = mean
            x_zero = stdv

        # use scipy to do the heavy lifting
        self.dist_obj = rayleigh(loc=x_zero, scale=a)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "ShiftedRayleigh"
