#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import expon

from .distribution import Distribution, _uses_native_parameters

__all__ = ["ShiftedExponential"]


class ShiftedExponential(Distribution):
    """Shifted exponential distribution.

    Supply either mean and std or ``rate`` and ``shift``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    rate : float, optional
        Positive rate parameter (the reciprocal of scale). Supply with the other native parameters instead of mean and std.
    shift : float, optional
        Additive shift, equal to the lower bound of the distribution. Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
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
