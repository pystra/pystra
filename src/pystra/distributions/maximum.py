#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .distribution import Distribution
from ._moments import _quantile_moments
from ..errors import ModelError

__all__ = ["Maximum"]


class Maximum(Distribution):
    """Distribution of maxima from the passed in parent distribution.

    Moments are computed by deterministic quantile integration, with absolute
    and relative tolerances of 1e-8 in standardized units. Nonfinite quantiles
    or unmet integration tolerance raise :class:`~pystra.ModelError`.

    :Attributes:
      - name (str):             Name of the random variable\n
      - mean (float):           Mean\n
      - std (float):           Standard deviation\n
      - parent (Distribution):  Parent distribution object
      - N (float):              Power to which distribution is raised
      - start_point (float):     Start point for seach\n
    """

    def __init__(self, name, parent, N, *, start_point=None):
        if not isinstance(parent, Distribution):
            raise ModelError(
                f"Maximum parent requires input of type {type(Distribution)}"
            )
        if not np.isfinite(N) or N < 1.0:
            raise ModelError("Maximum exponent must be >= 1.0")

        self.parent = parent
        self.N = N
        m, s = self._get_stats()

        super().__init__(
            name=name,
            mean=m,
            std=s,
            start_point=start_point,
        )

        self.dist_type = "Maximum"

    def pdf(self, x):
        """
        Probability density function
        """
        pdf = self.parent.pdf(x)
        cdf = 1.0
        if self.N > 1.0:
            cdf = self.parent.cdf(x)
        p = self.N * pdf * cdf ** (self.N - 1)
        return p

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        P = (self.parent.cdf(x)) ** self.N
        return P

    def ppf(self, p):
        """
        inverse cumulative distribution function
        """
        scalar_input = np.isscalar(p)
        x = self.parent.ppf(np.asarray(p) ** (1 / self.N))
        if scalar_input:
            return np.asarray(x).item()
        return x

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        p = self.std_normal.cdf(u)
        x = self.ppf(p)
        return x

    def x_to_u(self, x):
        """
        Transformation from x to u
        """
        u = self.std_normal.ppf(self.cdf(x))
        return u

    def jacobian(self, u, x):
        """
        Compute the Jacobian (e.g. Lemaire, eq. 4.9)
        """
        pdf1 = self.pdf(x)
        pdf2 = self.std_normal.pdf(u)
        J = np.diag(pdf1 / pdf2)
        return J

    def _get_stats(self):
        """Compute moments deterministically with quantile integration."""
        if self.N == 1:
            return self.parent.mean, self.parent.std
        return _quantile_moments(self, self.parent.mean, self.parent.std)

    def set_location(self, loc=0):
        """
        Updating the parent distribution location parameter.
        """
        self.parent.set_location(loc)
        self.update_stats()

    def set_scale(self, scale=1):
        """
        Updating the parent distribution scale parameter.
        """
        self.parent.set_scale(scale)
        self.update_stats()

    def update_stats(self):
        """
        Updates the mean and std estimates - used for sensitivity analysis
        where the parent distribution params may change after instantiation
        """
        m, s = self._get_stats()
        self._mean = m
        self._std = s
