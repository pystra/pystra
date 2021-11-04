#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import beta
import scipy.optimize as opt
from .distribution import Distribution


class Beta(Distribution):
    """Beta distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean or q\n
      - stdv (float): Standard deviation or r\n
      - a (float):    Lower boundary\n
      - b (float):    Uper boundary\n
      - input_type (any): Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, a=0, b=1, input_type=None, startpoint=None):

        if input_type is None:
            a = a
            b = b
            parameter_guess = 1
            par = opt.fmin(
                self.beta_parameter,
                parameter_guess,
                args=(a, b, mean, stdv),
                disp=False,
            )
            q = par[0]
            r = q * (b - a) * (mean - a) ** (-1) - q
        else:
            q = mean
            r = stdv
            a = a
            b = b

        # Use scipy for heavy lifting
        self.dist_obj = beta(q, r, loc=a, scale=b - a)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "Beta"

    def beta_parameter(self, q, *args):
        a, b, mean, stdv = args
        r = (b - mean) * (mean - a) ** (-1) * q
        f = np.absolute(
            ((b - a) * (q + r) ** (-1)) * (q * r * (q + r + 1) ** (-1)) ** 0.5 - stdv
        )
        return f
