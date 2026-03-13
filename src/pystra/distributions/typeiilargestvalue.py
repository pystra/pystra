#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import invweibull as frechet
import scipy.optimize as opt
import scipy.special as spec
from .distribution import Distribution


class TypeIIlargestValue(Distribution):
    """Type II largest value distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean or u_n\n
      - stdv (float): Standard deviation or k\n
      - input_type (any): Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):
        if input_type is None:
            parameter_guess = [2.000001]
            par = opt.fsolve(
                self.typIIlargest_parameter,
                parameter_guess,
                args=(mean, stdv),
            )
            k = par[0]
            u_n = mean / (spec.gamma(1 - 1 / k))
        else:
            u_n = mean
            k = stdv

        # Fréchet CDF: F(x) = exp(-(x/u_n)^{-k}), x > 0.
        # SciPy invweibull CDF: F(x) = exp(-((x-loc)/scale)^{-c}).
        # Direct mapping: c = k, loc = 0, scale = u_n.
        self.dist_obj = frechet(c=k, loc=0, scale=u_n)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "TypeIIlargestValue"

    def typIIlargest_parameter(self, x, *args):
        mean, stdv = args
        f = (spec.gamma(1 - 2 / x) - (spec.gamma(1 - 1 / x)) ** 2) ** 0.5 - (
            stdv / mean
        ) * spec.gamma(1 - 1 / x)
        return f
