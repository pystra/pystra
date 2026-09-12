"""Deterministic moments for distributions defined by a quantile function."""

import numpy as np
from scipy.integrate import quad

from ..errors import ModelError

__all__ = []


def _quantile_moments(distribution, center, scale):
    """Integrate standardized quantiles to avoid cancellation in the variance.

    Both integrals use absolute and relative tolerances of 1e-8 in
    standardized units, with at most 200 adaptive subintervals. Nonfinite
    integrands and unmet tolerances reject the distribution specification.
    """

    def integrate(function):
        value, error, *diagnostics = quad(
            function,
            0.0,
            1.0,
            epsabs=1e-8,
            epsrel=1e-8,
            limit=200,
            full_output=1,
        )
        if (
            len(diagnostics) != 1
            or not np.isfinite(value)
            or error > max(1e-8, 1e-8 * abs(value))
        ):
            raise ModelError(
                f"{type(distribution).__name__} moments did not meet integration tolerance"
            )
        return value

    def quantile(p):
        value = (float(distribution.ppf(p)) - center) / scale
        if not np.isfinite(value):
            raise ModelError(
                f"{type(distribution).__name__} has a nonfinite quantile during moment integration"
            )
        return value

    mean = integrate(quantile)
    variance = integrate(lambda p: (quantile(p) - mean) ** 2)
    if variance <= 0:
        raise ModelError("Integrated distribution variance must be positive")
    return center + scale * mean, scale * np.sqrt(variance)
