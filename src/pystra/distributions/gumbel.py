#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import gumbel_l, gumbel_r as gumbel

from .distribution import Distribution, _log1mexp, _uses_native_parameters

__all__ = ["Gumbel", "GumbelMin"]


class Gumbel(Distribution):
    """Gumbel distribution for maxima: the Type I extreme value distribution.

    :Attributes:
        - name (str):     Name of the random variable\n
        - mean (float): Mean\n
        - std (float): Standard deviation\n
        - loc (float): Location, given instead of mean and std\n
        - scale (float): Scale, given instead of mean and std\n
        - start_point (float): Start point for seach\n
    """

    def __init__(
        self, name, mean=None, std=None, *, loc=None, scale=None, start_point=None
    ):
        if _uses_native_parameters(self, mean, std, loc=loc, scale=scale):
            mu = loc
        else:
            mu = mean - 0.5772156649 * std * np.sqrt(6) / np.pi
            scale = std * np.sqrt(6) / np.pi

        # use scipy to do the heavy lifting
        self.dist_obj = gumbel(loc=mu, scale=scale)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Gumbel"

    def _upper_logsf(self, x):
        # 1 - F(x) = -expm1(-exp(-z)), whose logarithm is -z once exp(-z) < e**-40
        loc, scale = self.dist_obj.kwds["loc"], self.dist_obj.kwds["scale"]
        z = (np.asarray(x, dtype=float) - loc) / scale
        with np.errstate(over="ignore", under="ignore"):
            return np.where(z > 40.0, -z, _log1mexp(-np.exp(-z)))[()]

    def _lower_quantile_log(self, logp):
        # F(x) = exp(-exp(-(x - loc) / scale))
        loc, scale = self.dist_obj.kwds["loc"], self.dist_obj.kwds["scale"]
        with np.errstate(divide="ignore"):
            return loc - scale * np.log(-np.asarray(logp, dtype=float))

    def _upper_quantile_log(self, logq):
        # -log F = -log1p(-q), which equals q to double precision below e**-40
        loc, scale = self.dist_obj.kwds["loc"], self.dist_obj.kwds["scale"]
        logq = np.asarray(logq, dtype=float)
        with np.errstate(divide="ignore", under="ignore"):
            log_minus_log_f = np.where(
                logq < -40.0, logq, np.log(-np.log1p(-np.exp(logq)))
            )
        return loc - scale * log_minus_log_f


class GumbelMin(Distribution):
    """Gumbel distribution for minima: the Type I smallest value distribution.

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean\n
      - std (float): Standard deviation\n
      - loc (float): Location, given instead of mean and std\n
      - scale (float): Scale, given instead of mean and std\n
      - start_point (float): Start point for seach\n
    """

    def __init__(
        self, name, mean=None, std=None, *, loc=None, scale=None, start_point=None
    ):
        if _uses_native_parameters(self, mean, std, loc=loc, scale=scale):
            mu = loc
        else:
            beta = np.pi / (std * np.sqrt(6))
            mu = mean + (0.5772156649 * std * np.sqrt(6)) / np.pi
            scale = 1 / beta

        # use scipy to do the heavy lifting
        self.dist_obj = gumbel_l(loc=mu, scale=scale)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "GumbelMin"

    def _lower_logcdf(self, x):
        # F(x) = -expm1(-exp(z)), whose logarithm is z once exp(z) < e**-40
        loc, scale = self.dist_obj.kwds["loc"], self.dist_obj.kwds["scale"]
        z = (np.asarray(x, dtype=float) - loc) / scale
        with np.errstate(over="ignore", under="ignore"):
            return np.where(z < -40.0, z, _log1mexp(-np.exp(z)))[()]

    def _lower_quantile_log(self, logp):
        # 1 - F(x) = exp(-exp((x - loc) / scale)), and -log1p(-p) = p below e**-40
        loc, scale = self.dist_obj.kwds["loc"], self.dist_obj.kwds["scale"]
        logp = np.asarray(logp, dtype=float)
        with np.errstate(divide="ignore", under="ignore"):
            log_minus_log_s = np.where(
                logp < -40.0, logp, np.log(-np.log1p(-np.exp(logp)))
            )
        return loc + scale * log_minus_log_s

    def _upper_quantile_log(self, logq):
        loc, scale = self.dist_obj.kwds["loc"], self.dist_obj.kwds["scale"]
        with np.errstate(divide="ignore"):
            return loc + scale * np.log(-np.asarray(logq, dtype=float))
