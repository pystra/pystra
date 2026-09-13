#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import weibull_min as weibull
import scipy.optimize as opt
import scipy.special as spec
from .distribution import Distribution, _uses_native_parameters

__all__ = ["Weibull"]


class Weibull(Distribution):
    """Weibull distribution: the Type III extreme value distribution for minima.

    :Attributes:
        - name (str):       Name of the random variable\n
        - mean (float):     Mean\n
        - std (float):     Standard deviation\n
        - lower (float):    Lower bound\n
        - scale (float): Scale, measured from the lower bound, given instead of mean and std\n
        - shape (float): Shape, given instead of mean and std\n
        - start_point (float): Start point for seach\n
    """

    _native_parameters = ("scale", "shape")

    def _parameter_values(self):
        return {
            "scale": self.dist_obj.kwds["scale"],
            "shape": self.dist_obj.kwds["c"],
            "lower": self.lower,
        }

    def __init__(
        self,
        name,
        mean=None,
        std=None,
        *,
        scale=None,
        shape=None,
        lower=0,
        start_point=None,
    ):
        self.lower = lower
        epsilon = lower

        if not _uses_native_parameters(self, mean, std, scale=scale, shape=shape):
            meaneps = mean - epsilon
            parameter_guess = [0.1]
            par = opt.fsolve(
                self.weibull_parameter,
                parameter_guess,
                args=(meaneps, std),
            )
            k = par[0]
            u_1 = meaneps / (spec.gamma(1 + 1 / k)) + epsilon
            scale = u_1 - epsilon
        else:
            k = shape

        # use scipy to do the heavy lifting
        self.dist_obj = weibull(c=k, loc=epsilon, scale=scale)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Weibull"

    def weibull_parameter(self, x, *args):
        meaneps, std = args
        f = (spec.gamma(1 + 2 / x) - (spec.gamma(1 + 1 / x)) ** 2) ** 0.5 - (
            std / meaneps
        ) * spec.gamma(1 + 1 / x)
        return f
