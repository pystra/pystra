#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import ArrayLike
from scipy import special as sp
from scipy.stats import gumbel_l, gumbel_r as gumbel

from .distribution import Distribution, _log1mexp, _uses_native_parameters

__all__ = ["Gumbel", "GumbelMin"]


class Gumbel(Distribution):
    """Gumbel distribution for maxima: the Type I extreme value distribution.

    Supply either mean and std or ``loc`` and ``scale``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    loc : float, optional
        Location parameter. Supply with the other native parameters instead of mean and std.
    scale : float, optional
        Positive scale parameter. Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ("loc", "scale")

    def _parameter_values(self):
        return {"loc": self.dist_obj.kwds["loc"], "scale": self.dist_obj.kwds["scale"]}

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

    def u_to_x(self, u: ArrayLike) -> float | np.ndarray:
        """Map normal coordinates to Gumbel quantiles using logarithmic tails.

        Parameters
        ----------
        u : array_like
            Standard normal coordinates, as a scalar or an array.

        Returns
        -------
        float or ndarray
            Physical quantiles with the same shape as ``u``. Log probabilities
            retain finite quantiles beyond normal-probability underflow.
        """
        if type(self) is not Gumbel:
            # Preserve quantile overrides supplied by extension distributions.
            return super().u_to_x(u)
        u = np.asarray(u, dtype=float)
        loc, scale = self.dist_obj.kwds["loc"], self.dist_obj.kwds["scale"]
        if u.ndim == 0:
            value = float(u)
            if value > 3.0:
                logq = sp.log_ndtr(-value)
                term = logq if logq < -40.0 else np.log(-np.log1p(-sp.ndtr(-value)))
            elif value < -5.0:
                term = np.log(-sp.log_ndtr(value))
            else:
                term = np.log(-np.log(sp.ndtr(value)))
            return loc - scale * term
        shape = u.shape
        flat = u.ravel()
        upper = flat > 3.0
        lower = flat < -5.0
        central = ~(upper | lower)
        values = np.empty(flat.shape)
        values[central] = loc - scale * np.log(-np.log(sp.ndtr(flat[central])))
        if lower.any():
            values[lower] = self._lower_quantile_log(sp.log_ndtr(flat[lower]))
        if upper.any():
            values[upper] = self._upper_quantile_log(sp.log_ndtr(-flat[upper]))
        return values.reshape(shape)

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

    Supply either mean and std or ``loc`` and ``scale``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    loc : float, optional
        Location parameter. Supply with the other native parameters instead of mean and std.
    scale : float, optional
        Positive scale parameter. Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ("loc", "scale")

    def _parameter_values(self):
        return {"loc": self.dist_obj.kwds["loc"], "scale": self.dist_obj.kwds["scale"]}

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
