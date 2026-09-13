#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import beta
import scipy.optimize as opt
from .distribution import Distribution, _uses_native_parameters

__all__ = ["Beta"]


class Beta(Distribution):
    """Beta distribution on a bounded interval.

    Supply either mean and std or ``q`` and ``r``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    q : float, optional
        First beta shape parameter. Supply with the other native parameters instead of mean and std.
    r : float, optional
        Second beta shape parameter. Supply with the other native parameters instead of mean and std.
    lower : float, optional
        Lower bound of the distribution. Defaults to 0.
    upper : float, optional
        Upper bound of the distribution. Defaults to 1.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ("q", "r")

    def _parameter_values(self):
        return {
            "q": self.dist_obj.args[0],
            "r": self.dist_obj.args[1],
            "lower": self.lower,
            "upper": self.upper,
        }

    def __init__(
        self,
        name,
        mean=None,
        std=None,
        *,
        q=None,
        r=None,
        lower=0,
        upper=1,
        start_point=None,
    ):
        self.lower = lower
        self.upper = upper
        a = lower
        b = upper

        if not _uses_native_parameters(self, mean, std, q=q, r=r):
            parameter_guess = 1
            par = opt.fmin(
                self.beta_parameter,
                parameter_guess,
                args=(a, b, mean, std),
                disp=False,
            )
            q = par[0]
            r = q * (b - a) * (mean - a) ** (-1) - q

        # Use scipy for heavy lifting
        self.dist_obj = beta(q, r, loc=a, scale=b - a)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Beta"

    def beta_parameter(self, q, *args):
        a, b, mean, std = args
        r = (b - mean) * (mean - a) ** (-1) * q
        f = np.absolute(
            ((b - a) * (q + r) ** (-1)) * (q * r * (q + r + 1) ** (-1)) ** 0.5 - std
        )
        return f
