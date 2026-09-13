#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rayleigh

from .distribution import Distribution, _uses_native_parameters

__all__ = ["ShiftedRayleigh"]


class ShiftedRayleigh(Distribution):
    """Shifted Rayleigh distribution.

    Supply either mean and std or ``scale`` and ``shift``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    scale : float, optional
        Positive scale parameter. Supply with the other native parameters instead of mean and std.
    shift : float, optional
        Additive shift, equal to the lower bound of the distribution. Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ("scale", "shift")

    def _parameter_values(self):
        return {
            "scale": self.dist_obj.kwds["scale"],
            "shift": self.dist_obj.kwds["loc"],
        }

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
