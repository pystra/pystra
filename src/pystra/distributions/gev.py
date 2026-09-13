"""Generalized extreme value distributions for maxima and minima."""

from typing import overload

from numpy.typing import ArrayLike
import numpy as np
from scipy.stats import genextreme
from scipy.special import gamma

from pystra.distributions import Distribution
from .distribution import _uses_native_parameters

__all__ = ["GEV", "GEVmax", "GEVMin"]


class GEV(Distribution):
    """Generalized extreme value (GEV) distribution for maxima.

    GEVmax is an alias for this class.

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
    shape : float
        Required shape parameter, less than 0.5 for finite variance.
        The displayed None default marks a missing argument and is rejected.
        Negative, zero and positive values give the Weibull, Gumbel and
        Fréchet cases, respectively.
    loc : float, optional
        Location parameter. Supply with the other native parameters instead of mean and std.
    scale : float, optional
        Positive scale parameter. Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.

    Raises
    ------
    ValueError
        If shape is greater than or equal to 0.5.
    """

    _native_parameters = ("loc", "scale")

    def _parameter_values(self):
        return {
            "shape": self.shape,
            "loc": self.dist_obj.kwds["loc"],
            "scale": self.dist_obj.kwds["scale"],
        }

    @overload
    def __init__(
        self,
        name: str,
        mean: float,
        std: float,
        shape: float,
        *,
        loc: None = None,
        scale: None = None,
        start_point: float | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        mean: None = None,
        std: None = None,
        *,
        shape: float,
        loc: float,
        scale: float,
        start_point: float | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        mean: None,
        std: None,
        shape: float,
        *,
        loc: float,
        scale: float,
        start_point: float | None = None,
    ) -> None: ...

    def __init__(
        self,
        name: str,
        mean: float | None = None,
        std: float | None = None,
        shape: float | None = None,
        *,
        loc: float | None = None,
        scale: float | None = None,
        start_point: float | None = None,
    ) -> None:
        if shape is None:
            raise TypeError(f"{type(self).__name__} needs shape")
        if shape >= 0.5:
            raise ValueError("`shape` must be less than 0.5 for finite variance")

        self.shape = shape

        g1 = gamma(1 - shape)
        g2 = gamma(1 - 2 * shape)

        if not _uses_native_parameters(self, mean, std, loc=loc, scale=scale):
            if np.isclose(shape, 0):
                scale = std * np.sqrt(6) / np.pi
                loc = mean - scale * np.euler_gamma
            else:
                scale = std * np.abs(shape) / np.sqrt(g2 - g1**2)
                loc = mean - scale / shape * (g1 - 1)
        else:
            if np.isclose(shape, 0):
                self._mean = loc + scale * np.euler_gamma
                self._std = scale * np.pi / np.sqrt(6)
            else:
                self._mean = loc + (g1 - 1) * scale / shape
                self._std = np.sqrt((g2 - g1**2) * (scale / shape) ** 2)

        # use scipy to do the heavy lifting
        self.dist_obj = genextreme(c=-shape, loc=loc, scale=scale)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "GEV"

    @property
    def sensitivity_params(self) -> dict[str, float]:
        r"""Sensitivity parameters for GEV.

        Returns ``{"mean": μ, "std": σ, "shape": ξ}``.  The shape
        parameter ξ controls tail behavior: ξ < 0 is Weibull (bounded
        upper tail), ξ = 0 is Gumbel, ξ > 0 is Fréchet (heavy-tailed).
        """
        return {"mean": self.mean, "std": self.std, "shape": self.shape}


# ``GEVmax`` is kept as another name for ``GEV``: the one deliberate alias in
# the 2.0 API, at the maintainer's request.
GEVmax = GEV


class GEVMin(Distribution):
    """Generalized extreme value (GEV) distribution for minima.

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
    shape : float
        Required shape parameter, less than 0.5 for finite variance.
        The displayed None default marks a missing argument and is rejected.
        Negative, zero and positive values give the Weibull, Gumbel and
        Fréchet cases, respectively.
    loc : float, optional
        Location parameter. Supply with the other native parameters instead of mean and std.
    scale : float, optional
        Positive scale parameter. Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.

    Raises
    ------
    ValueError
        If shape is greater than or equal to 0.5.
    """

    _native_parameters = ("loc", "scale")

    def _parameter_values(self):
        return {
            "shape": self.shape,
            "loc": -self.dist_obj.kwds["loc"],
            "scale": self.dist_obj.kwds["scale"],
        }

    @overload
    def __init__(
        self,
        name: str,
        mean: float,
        std: float,
        shape: float,
        *,
        loc: None = None,
        scale: None = None,
        start_point: float | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        mean: None = None,
        std: None = None,
        *,
        shape: float,
        loc: float,
        scale: float,
        start_point: float | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        mean: None,
        std: None,
        shape: float,
        *,
        loc: float,
        scale: float,
        start_point: float | None = None,
    ) -> None: ...

    def __init__(
        self,
        name: str,
        mean: float | None = None,
        std: float | None = None,
        shape: float | None = None,
        *,
        loc: float | None = None,
        scale: float | None = None,
        start_point: float | None = None,
    ) -> None:
        if shape is None:
            raise TypeError(f"{type(self).__name__} needs shape")
        if shape >= 0.5:
            raise ValueError("`shape` must be less than 0.5 for finite variance")

        self.shape = shape

        g1 = gamma(1 - shape)
        g2 = gamma(1 - 2 * shape)

        if not _uses_native_parameters(self, mean, std, loc=loc, scale=scale):
            # mean and std passed in
            self._mean = mean
            self._std = std
            if np.isclose(shape, 0):
                scale = self.std * np.sqrt(6) / np.pi
                loc = self.mean + scale * np.euler_gamma
            else:
                scale = self.std * np.abs(shape) / np.sqrt(g2 - g1**2)
                loc = self.mean + (scale / shape) * (g1 - 1)
        else:
            # loc and scale are actual GEV parameters
            if np.isclose(shape, 0):
                self._mean = loc - scale * np.euler_gamma
                self._std = scale * np.pi / np.sqrt(6)
            else:
                self._mean = loc - (g1 - 1) * scale / shape
                self._std = np.sqrt((g2 - g1**2) * (scale / shape) ** 2)

        # use scipy to do the heavy lifting; note reverse shape sign convention
        self.dist_obj = genextreme(c=-shape, loc=-loc, scale=scale)

        # Save correct moments before super().__init__() — the dist_obj
        # represents -X internally, so dist_obj.mean() gives the wrong
        # sign and would overwrite the correct values via _update_moments()
        _correct_mean = self.mean
        _correct_stdv = self.std

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        # Restore correct moments (dist_obj models -X internally)
        self._mean = _correct_mean
        self._std = _correct_stdv
        if start_point is None:
            self._start_point = self.mean

        self.dist_type = "GEVMin"

    @property
    def sensitivity_params(self) -> dict[str, float]:
        r"""Sensitivity parameters for GEVMin.

        Returns ``{"mean": μ, "std": σ, "shape": ξ}``.  The shape
        parameter ξ controls tail behavior: ξ < 0 is Weibull (bounded
        lower tail), ξ = 0 is Gumbel, ξ > 0 is Fréchet (heavy-tailed).
        """
        return {"mean": self.mean, "std": self.std, "shape": self.shape}

    def pdf(self, x: ArrayLike) -> float | np.ndarray:
        """Evaluate the probability density function."""
        return self.dist_obj.pdf(-np.asarray(x, dtype=float))

    def logpdf(self, x: ArrayLike) -> float | np.ndarray:
        """Log density."""
        return self.dist_obj.logpdf(-np.asarray(x, dtype=float))

    def cdf(self, x: ArrayLike) -> float | np.ndarray:
        """Evaluate the cumulative distribution function."""
        return self.dist_obj.sf(-np.asarray(x, dtype=float))

    def sf(self, x: ArrayLike) -> float | np.ndarray:
        """Survival function."""
        return self.dist_obj.cdf(-np.asarray(x, dtype=float))

    def _lower_logcdf(self, x):
        return self.dist_obj.logsf(-np.asarray(x, dtype=float))

    def _upper_logsf(self, x):
        return self.dist_obj.logcdf(-np.asarray(x, dtype=float))

    def ppf(self, u: ArrayLike) -> float | np.ndarray:
        """Evaluate the inverse cumulative distribution function."""
        return -self.dist_obj.isf(u)

    def isf(self, q: ArrayLike) -> float | np.ndarray:
        """Inverse survival function."""
        return -self.dist_obj.ppf(q)
