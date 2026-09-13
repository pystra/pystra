"""Distributions of independent maxima and their deterministic moments."""

from numpy.typing import ArrayLike
import numpy as np

from .distribution import Distribution, _log1mexp, _piecewise
from ._moments import _quantile_moments
from ..errors import ModelError

__all__ = ["Maximum"]


class Maximum(Distribution):
    """Distribution of maxima from a supplied parent distribution.

    Moments are computed deterministically by Gauss-Hermite quadrature of the
    quantile function in standard normal space, with exact identities for
    Frechet and uniform parents. Successive rules are compared at relative
    tolerance 1e-7 and absolute tolerance 1e-9 in standardized units; a miss
    warns and keeps the finest estimate, which serves for start points and
    finite-difference steps. Nonfinite moments raise :class:`~pystra.ModelError`.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    parent : Distribution
        Parent distribution whose CDF is raised to the power N.
    N : float
        Finite exponent, at least 1. Integer N represents N independent observations.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ()

    def _parameter_values(self):
        return {"parent": self.parent, "N": self.N}

    @property
    def sensitivity_params(self) -> dict[str, float]:
        """No generic moment perturbation; replace the constructor inputs."""
        return {}

    def __init__(
        self,
        name: str,
        parent: Distribution,
        N: float,
        *,
        start_point: float | None = None,
    ) -> None:
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

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the probability density function."""
        pdf = self.parent.pdf(x)
        cdf = 1.0
        if self.N > 1.0:
            cdf = self.parent.cdf(x)
        p = self.N * pdf * cdf ** (self.N - 1)
        return p

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the cumulative distribution function."""
        P = (self.parent.cdf(x)) ** self.N
        return P

    def ppf(self, p: ArrayLike) -> float | np.ndarray:
        """Evaluate the inverse cumulative distribution function."""
        with np.errstate(divide="ignore"):
            return self._ppf_log(np.log(np.asarray(p, dtype=float)))

    def isf(self, q: ArrayLike) -> float | np.ndarray:
        """Inverse survival function."""
        with np.errstate(divide="ignore"):
            return self._isf_log(np.log(np.asarray(q, dtype=float)))

    def logpdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Log density, ``log N + log f(x) + (N - 1) log F(x)``."""
        logpdf = np.log(self.N) + self.parent.logpdf(x)
        if self.N == 1:
            return logpdf
        return logpdf + (self.N - 1) * self.parent.logcdf(x)

    def logcdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Log CDF, ``N`` times the parent's."""
        return self.N * self.parent.logcdf(x)

    def sf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Survival function ``1 - F(x)**N``."""
        return -np.expm1(self.logcdf(x))

    def logsf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Log survival function."""
        # Once 1 - F**N is below 1e-200 it equals N (1 - F) to double precision
        a = np.asarray(self.logcdf(x), dtype=float)
        return _piecewise(
            x,
            a > -1e-200,
            lambda v: _log1mexp(self.logcdf(v)),
            lambda v: np.log(self.N) + self.parent.logsf(v),
        )

    def _lower_quantile_log(self, logp):
        # F(x)**N = p, so the parent's log CDF is log(p) / N
        return self.parent._ppf_log(np.asarray(logp, dtype=float) / self.N)

    def _upper_quantile_log(self, logq):
        # 1 - F(x)**N = q gives the parent survival -expm1(log1p(-q) / N),
        # which is q / N to double precision once q is below 1e-200
        logq = np.asarray(logq, dtype=float)
        with np.errstate(divide="ignore", under="ignore"):
            q = np.exp(logq)
            log_parent_sf = np.where(
                q < 1e-200, logq - np.log(self.N), _log1mexp(np.log1p(-q) / self.N)
            )
        return self.parent._isf_log(log_parent_sf)

    def _get_stats(self):
        """Compute moments deterministically with quantile integration."""
        if self.N == 1:
            return self.parent.mean, self.parent.std
        return _quantile_moments(self, self.parent.mean, self.parent.std)

    def set_location(self, loc: float = 0) -> None:
        """
        Updating the parent distribution location parameter.
        """
        self.parent.set_location(loc)
        self.update_stats()

    def set_scale(self, scale: float = 1) -> None:
        """
        Updating the parent distribution scale parameter.
        """
        self.parent.set_scale(scale)
        self.update_stats()

    def update_stats(self) -> None:
        """Recompute moments after an in-place change to the constructor inputs."""
        m, s = self._get_stats()
        self._mean = m
        self._std = s
