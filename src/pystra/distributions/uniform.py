"""Uniform marginal distribution."""

from numpy.typing import ArrayLike
import numpy as np
from scipy.stats import uniform

from .distribution import Distribution, _uses_native_parameters

__all__ = ["Uniform"]


class Uniform(Distribution):
    """Uniform distribution.

    Supply either mean and std or ``lower`` and ``upper``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    lower : float, optional
        Lower bound of the distribution. Supply with the other native parameters instead of mean and std.
    upper : float, optional
        Upper bound of the distribution. Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ("lower", "upper")

    def _parameter_values(self):
        return {"lower": self.a, "upper": self.b}

    def __init__(
        self,
        name: str,
        mean: float | None = None,
        std: float | None = None,
        *,
        lower: float | None = None,
        upper: float | None = None,
        start_point: float | None = None,
    ) -> None:
        if _uses_native_parameters(self, mean, std, lower=lower, upper=upper):
            a = lower
            b = upper
        else:
            a = mean - 3**0.5 * std
            b = mean + 3**0.5 * std

        self.a = a
        self.b = b

        # use scipy to do the heavy lifting
        self.dist_obj = uniform(loc=a, scale=b - a)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Uniform"

    # Overriding these for performance

    def u_to_x(self, u: ArrayLike) -> float | np.ndarray:
        """
        Transformation from u to x, measured from the nearer bound
        """
        u = np.asarray(u, dtype=float)
        tail = self.std_normal.cdf(-np.abs(u))
        width = self.b - self.a
        if u.ndim == 0:
            return self.a + width * tail if u <= 0 else self.b - width * tail
        return np.where(u <= 0, self.a + width * tail, self.b - width * tail)[()]
