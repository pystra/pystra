#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import beta
import scipy.optimize as opt
from .distribution import Distribution, _uses_native_parameters

__all__ = ["Beta"]


class Beta(Distribution):
    """Beta distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean\n
      - std (float): Standard deviation\n
      - lower (float): Lower bound\n
      - upper (float): Upper bound\n
      - q (float): First shape parameter, given instead of mean and std\n
      - r (float): Second shape parameter, given instead of mean and std\n
      - start_point (float): Start point for seach\n
    """

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
        self._ctor_kwargs = {"lower": lower, "upper": upper}
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
