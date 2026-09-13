#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import invweibull as frechet
import scipy.optimize as opt
import scipy.special as spec
from .distribution import Distribution, _uses_native_parameters

__all__ = ["Frechet"]


class Frechet(Distribution):
    """Fréchet distribution: the Type II extreme value distribution for maxima.

    Supply either mean and std or ``scale`` and ``shape``.
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
    shape : float, optional
        Shape parameter. Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ("scale", "shape")

    def _parameter_values(self):
        return {"scale": self.dist_obj.kwds["scale"], "shape": self.dist_obj.kwds["c"]}

    def __init__(
        self, name, mean=None, std=None, *, scale=None, shape=None, start_point=None
    ):
        if not _uses_native_parameters(self, mean, std, scale=scale, shape=shape):
            parameter_guess = [2.000001]
            par = opt.fsolve(
                self.frechet_parameter,
                parameter_guess,
                args=(mean, std),
            )
            k = par[0]
            u_n = mean / (spec.gamma(1 - 1 / k))
        else:
            u_n = scale
            k = shape

        # Fréchet CDF: F(x) = exp(-(x/u_n)^{-k}), x > 0.
        # SciPy invweibull CDF: F(x) = exp(-((x-loc)/scale)^{-c}).
        # Direct mapping: c = k, loc = 0, scale = u_n.
        self.dist_obj = frechet(c=k, loc=0, scale=u_n)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Frechet"

    def frechet_parameter(self, x, *args):
        mean, std = args
        f = (spec.gamma(1 - 2 / x) - (spec.gamma(1 - 1 / x)) ** 2) ** 0.5 - (
            std / mean
        ) * spec.gamma(1 - 1 / x)
        return f
