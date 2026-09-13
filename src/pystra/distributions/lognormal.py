"""Lognormal marginal distribution."""

from numpy.typing import ArrayLike
import numpy as np
from scipy import special as sp
from scipy.stats import lognorm

from .distribution import Distribution, _uses_native_parameters

__all__ = ["Lognormal"]


class Lognormal(Distribution):
    """Lognormal distribution using direct PDF and CDF formulas.

    Supply either mean and std or ``log_mean`` and ``log_std``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    log_mean : float, optional
        Mean of the underlying normal variable, log(X). Supply with the other native parameters instead of mean and std.
    log_std : float, optional
        Standard deviation of the underlying normal variable, log(X). Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ("log_mean", "log_std")

    def _parameter_values(self):
        return {"log_mean": self.lamb, "log_std": self.zeta}

    def __init__(
        self,
        name: str,
        mean: float | None = None,
        std: float | None = None,
        *,
        log_mean: float | None = None,
        log_std: float | None = None,
        start_point: float | None = None,
    ) -> None:
        if _uses_native_parameters(self, mean, std, log_mean=log_mean, log_std=log_std):
            self.lamb = log_mean
            self.zeta = log_std
        else:
            # infer parameters from the moments
            self._update_params(mean, std)

        # Could use scipy to do the heavy lifting. However, there is a small
        # performance hit, so for this common dist use bespoke implementation
        # for the PDF, CDF.
        # Careful: the scipy parametrization is tricky!
        self.dist_obj = lognorm(scale=np.exp(self.lamb), s=self.zeta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Lognormal"

    def _update_params(self, mean, std):
        cov = std / mean
        self.zeta = (np.log(1 + cov**2)) ** 0.5
        self.lamb = np.log(mean) - 0.5 * self.zeta**2

    # Overriding base class implementations for speed

    @property
    def _shift(self):
        """Lower bound of the support; :class:`ShiftedLognormal` moves it."""
        return 0.0

    def _z(self, x):
        """Standardized log value, ``-inf`` at or below the lower bound."""
        y = np.asarray(x, dtype=float) - self._shift
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.log(y) - self.lamb) / self.zeta
        return np.where(y <= 0, -np.inf, z)[()]

    def pdf(self, x: ArrayLike) -> float | np.ndarray:
        """Evaluate the probability density function."""
        y = np.asarray(x, dtype=float) - self._shift
        z = self._z(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * self.zeta * y)
        return np.where(y <= 0, 0.0, p)[()]

    def logpdf(self, x: ArrayLike) -> float | np.ndarray:
        """Log density."""
        y = np.asarray(x, dtype=float) - self._shift
        z = self._z(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = -0.5 * z**2 - np.log(self.zeta * y) - 0.5 * np.log(2 * np.pi)
        return np.where(y <= 0, -np.inf, lp)[()]

    def cdf(self, x: ArrayLike) -> float | np.ndarray:
        """Evaluate the cumulative distribution function."""
        return sp.ndtr(self._z(x))

    def sf(self, x: ArrayLike) -> float | np.ndarray:
        """Survival function."""
        return sp.ndtr(-self._z(x))

    def logcdf(self, x: ArrayLike) -> float | np.ndarray:
        """Log CDF."""
        return sp.log_ndtr(self._z(x))

    def logsf(self, x: ArrayLike) -> float | np.ndarray:
        """Log survival function."""
        return sp.log_ndtr(-self._z(x))

    def _lower_quantile_log(self, logp):
        return self._shift + np.exp(self.lamb + self.zeta * sp.ndtri_exp(logp))

    def _upper_quantile_log(self, logq):
        return self._shift + np.exp(self.lamb - self.zeta * sp.ndtri_exp(logq))

    def u_to_x(self, u: float | np.ndarray) -> float | np.ndarray:
        """Transform standard normal coordinates to physical values."""
        x = self._shift + np.exp(u * self.zeta + self.lamb)
        return x

    def x_to_u(self, x: ArrayLike) -> float | np.ndarray:
        """Transform physical values to standard normal coordinates."""
        return self._z(x)

    def cdf_gradient(self, x: ArrayLike) -> dict[str, float | np.ndarray]:
        r"""Analytical derivatives of the Lognormal CDF w.r.t. μ and σ.

        The CDF is ``F(x) = Φ((ln x - λ) / ζ)`` where
        ``ζ = sqrt(ln(1 + (σ/μ)²))`` and ``λ = ln(μ) - ζ²/2``.

        The chain rule gives:

        .. math::
            \frac{\partial F}{\partial \theta}
            = \frac{\varphi(z)}{\zeta}
              \left(-\frac{\partial\lambda}{\partial\theta}
                    - z\,\frac{\partial\zeta}{\partial\theta}\right)

        where ``z = (ln x - λ) / ζ``.
        """
        cov = self.std / self.mean
        cov2 = cov**2
        z = (np.log(x) - self.lamb) / self.zeta
        phi_z = self.std_normal.pdf(z)

        # Derivatives of ζ and λ w.r.t. μ and σ
        # ζ² = ln(1 + cov²),  cov = σ/μ
        # ∂ζ/∂μ = (1/ζ) × (1/(1+cov²)) × (-cov²/μ) = -cov² / (μ ζ (1+cov²))
        # ∂ζ/∂σ = (1/ζ) × (1/(1+cov²)) × (cov/μ)   =  cov  / (μ ζ (1+cov²))
        dzeta_dmu = -cov2 / (self.mean * self.zeta * (1 + cov2))
        dzeta_dsig = cov / (self.mean * self.zeta * (1 + cov2))

        # λ = ln(μ) - ζ²/2
        # ∂λ/∂μ = 1/μ - ζ ∂ζ/∂μ
        # ∂λ/∂σ = -ζ ∂ζ/∂σ
        dlamb_dmu = 1.0 / self.mean - self.zeta * dzeta_dmu
        dlamb_dsig = -self.zeta * dzeta_dsig

        # ∂F/∂θ = (φ(z)/ζ) × (-∂λ/∂θ - z ∂ζ/∂θ)
        coeff = phi_z / self.zeta
        dF_dmu = coeff * (-dlamb_dmu - z * dzeta_dmu)
        dF_dsig = coeff * (-dlamb_dsig - z * dzeta_dsig)

        return {"mean": dF_dmu, "std": dF_dsig}

    def set_location(self, loc: float = 0) -> None:
        """Set the physical mean to loc and update the underlying normal parameters."""

        self._update_params(loc, self.std)
        self._mean = loc

    def set_scale(self, scale: float = 1) -> None:
        """Set the physical standard deviation to scale and update normal parameters."""
        self._update_params(self.mean, scale)
        self._std = scale
