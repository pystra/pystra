"""Deterministic moments for distributions defined by a quantile function."""

from functools import lru_cache
import warnings

import numpy as np
from scipy.special import log_ndtr, roots_hermitenorm

from ..errors import ModelError

__all__ = []


@lru_cache(maxsize=4)
def _normal_rule(order):
    nodes, weights = roots_hermitenorm(order)
    return nodes, weights / np.sqrt(2 * np.pi)


def _quantile_moments(distribution, center, scale):
    """Integrate standardized quantiles against the standard normal density.

    Vectorized Gauss-Hermite rules remove the endpoint singularities of
    integration in probability space. Orders 32, 64, 128 and 256 are compared
    at absolute tolerance 1e-9 and relative tolerance 1e-7 in standardized
    mean/variance. An unmet tolerance warns and returns the finest estimate:
    these moments set start points and finite-difference steps, and a missed
    numerical tolerance does not establish that a distribution is invalid.
    The log-quantile contract evaluates x(u) without redundant tail-validation
    root solves; finite endpoints are approximated within 1e-12 parent std.
    Nonfinite estimates or nonpositive variance still raise ModelError.
    """
    # Powers of a Frechet CDF only rescale it. Numerical quadrature would
    # converge arbitrarily slowly as the shape approaches two (finite variance).
    parent = getattr(distribution, "parent", None)
    exponent = distribution.N
    if parent is None:
        parent = distribution.max_dist
        exponent = 1 / exponent
    frozen = getattr(parent, "dist_obj", None)
    if frozen is not None and frozen.dist.name == "invweibull":
        shape = frozen.kwds["c"]
        factor = exponent ** (1 / shape)
        loc = frozen.kwds.get("loc", 0)
        return loc + (center - loc) * factor, scale * factor

    if frozen is not None and frozen.dist.name == "uniform":
        lo, hi = frozen.support()
        mean = lo + (hi - lo) * exponent / (exponent + 1)
        std = (hi - lo) * np.sqrt(exponent / (exponent + 2)) / (exponent + 1)
        return mean, std

    # Once a parent quantile is within 1e-12 parent standard deviations of
    # its finite lower endpoint, use that endpoint. Its standardized error
    # is bounded by 1e-12; solving log-CDFs for still smaller values only
    # adds work (often hundreds of bisections) and no useful moment accuracy.
    lower = float(parent.ppf(0.0))
    log_bound = -np.inf
    if np.isfinite(lower):
        log_bound = float(parent.logcdf(lower + scale * 1e-12))
    previous = None
    for order in (32, 64, 128, 256):
        nodes, weights = _normal_rule(order)
        logp = log_ndtr(nodes)
        at_bound = logp / exponent < log_bound
        quantiles = np.full(nodes.shape, lower)
        quantiles[~at_bound] = distribution._ppf_log(logp[~at_bound])
        standardized = (quantiles - center) / scale
        if not np.all(np.isfinite(standardized)):
            raise ModelError(
                f"{type(distribution).__name__} has a nonfinite quantile during moment integration"
            )
        mean = float(weights @ standardized)
        variance = float(np.sum((np.sqrt(weights) * (standardized - mean)) ** 2))
        estimate = np.array([mean, variance])
        if not np.all(np.isfinite(estimate)) or variance <= 0:
            raise ModelError(
                "Integrated distribution variance must be positive and finite"
            )
        if previous is not None and np.allclose(
            estimate, previous, rtol=1e-7, atol=1e-9
        ):
            return center + scale * mean, scale * np.sqrt(variance)
        previous = estimate

    warnings.warn(
        f"{type(distribution).__name__} moment quadrature did not reach tolerance; "
        "using the finest finite estimate for start points and differentiation",
        RuntimeWarning,
        stacklevel=2,
    )
    return center + scale * mean, scale * np.sqrt(variance)
