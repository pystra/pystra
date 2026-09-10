#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import weibull_min as weibull
import scipy.optimize as opt
import scipy.special as spec
from .distribution import Distribution

__all__ = ["Weibull"]


class Weibull(Distribution):
    """Weibull distribution: the Type III extreme value distribution for minima.

    :Attributes:
        - name (str):       Name of the random variable\n
        - mean (float):     Mean or u_1\n
        - std (float):     Standard deviation or k\n
        - epsilon (float):  Epsilon\n
        - input_type (any): Change meaning of mean and std\n
        - start_point (float): Start point for seach\n
    """

    def __init__(self, name, mean, std, epsilon=0, input_type=None, start_point=None):
        self.epsilon = epsilon
        self._ctor_kwargs = {"epsilon": epsilon}

        if input_type is None:
            mean = mean
            std = std
            epsilon = epsilon
            meaneps = mean - epsilon
            parameter_guess = [0.1]
            par = opt.fsolve(
                self.weibull_parameter,
                parameter_guess,
                args=(meaneps, std),
            )
            k = par[0]
            u_1 = meaneps / (spec.gamma(1 + 1 / k)) + epsilon
        else:
            u_1 = mean
            k = std
            epsilon = epsilon

        # use scipy to do the heavy lifting
        self.dist_obj = weibull(c=k, loc=epsilon, scale=u_1 - epsilon)

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
