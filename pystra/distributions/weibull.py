#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import weibull_min as weibull
import scipy.optimize as opt
import scipy.special as spec
from .distribution import Distribution


class Weibull(Distribution):
    """Weibull distribution

    :Attributes:
        - name (str):       Name of the random variable\n
        - mean (float):     Mean or u_1\n
        - stdv (float):     Standard deviation or k\n
        - epsilon (float):  Epsilon\n
        - input_type (any): Change meaning of mean and stdv\n
        - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, epsilon=0, input_type=None, startpoint=None):

        if input_type is None:
            mean = mean
            stdv = stdv
            epsilon = epsilon
            meaneps = mean - epsilon
            parameter_guess = [0.1]
            par = opt.fsolve(
                self.weibull_parameter,
                parameter_guess,
                args=(meaneps, stdv),
            )
            k = par[0]
            u_1 = meaneps / (spec.gamma(1 + 1 / k)) + epsilon
        else:
            u_1 = mean
            k = stdv
            epsilon = epsilon

        # use scipy to do the heavy lifting
        self.dist_obj = weibull(c=k, loc=epsilon, scale=u_1 - epsilon)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "Weibull"

    def weibull_parameter(self, x, *args):
        meaneps, stdv = args
        f = (spec.gamma(1 + 2 / x) - (spec.gamma(1 + 1 / x)) ** 2) ** 0.5 - (
            stdv / meaneps
        ) * spec.gamma(1 + 1 / x)
        return f
