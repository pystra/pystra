#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import expon

from .distribution import Distribution, _uses_native_parameters

__all__ = ["ShiftedExponential"]


class ShiftedExponential(Distribution):
    """Shifted exponential distribution

    :Attributes:
        - name (str):         Name of the random variable\n
        - mean (float):       Mean\n
        - std (float):       Standard deviation\n
        - rate (float): Rate, given instead of mean and std\n
        - shift (float): Lower bound, given instead of mean and std\n
        - start_point (float): Start point for seach\n
    """

    _native_parameters = ("rate", "shift")

    def _parameter_values(self):
        return {
            "rate": 1 / self.dist_obj.kwds["scale"],
            "shift": self.dist_obj.kwds["loc"],
        }

    def __init__(
        self, name, mean=None, std=None, *, rate=None, shift=None, start_point=None
    ):
        if _uses_native_parameters(self, mean, std, rate=rate, shift=shift):
            lamb = rate
            x_zero = shift
        else:
            x_zero = mean - std
            lamb = 1 / std

        # use scipy to do the heavy lifting
        self.dist_obj = expon(loc=x_zero, scale=1 / lamb)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "ShiftedExponential"
