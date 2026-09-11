#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rayleigh

from .distribution import Distribution, _uses_native_parameters

__all__ = ["ShiftedRayleigh"]


class ShiftedRayleigh(Distribution):
    """Shifted Rayleigh distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean\n
      - std (float): Standard deviation\n
      - scale (float): Scale, given instead of mean and std\n
      - shift (float): Lower bound, given instead of mean and std\n
      - start_point (float): Start point for seach\n
    """

    def __init__(
        self, name, mean=None, std=None, *, scale=None, shift=None, start_point=None
    ):
        if _uses_native_parameters(self, mean, std, scale=scale, shift=shift):
            a = scale
            x_zero = shift
        else:
            a = std / ((2 - np.pi * 0.5) ** 0.5)
            x_zero = mean - std * (np.pi / (4 - np.pi)) ** 0.5

        # use scipy to do the heavy lifting
        self.dist_obj = rayleigh(loc=x_zero, scale=a)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "ShiftedRayleigh"
