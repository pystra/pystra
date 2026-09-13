"""Weibull marginal distribution."""

import numpy as np
from scipy.stats import weibull_min as weibull
import scipy.optimize as opt
import scipy.special as spec

from .distribution import Distribution, _uses_native_parameters

__all__ = ["Weibull"]


class Weibull(Distribution):
    """Weibull distribution: the Type III extreme value distribution for minima.

    Supply either mean and std or ``scale`` and ``shape``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    scale : float, optional
        Positive scale parameter. Supply with the other native parameters instead of mean and std.
    shape : float, optional
        Shape parameter. Supply with the other native parameters instead of mean and std.
    lower : float, optional
        Lower bound of the distribution. Defaults to 0.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ("scale", "shape")

    def _parameter_values(self):
        return {
            "scale": self.dist_obj.kwds["scale"],
            "shape": self.dist_obj.kwds["c"],
            "lower": self.lower,
        }

    def __init__(
        self,
        name: str,
        mean: float | None = None,
        std: float | None = None,
        *,
        scale: float | None = None,
        shape: float | None = None,
        lower: float = 0,
        start_point: float | None = None,
    ) -> None:
        self.lower = lower
        epsilon = lower

        if not _uses_native_parameters(self, mean, std, scale=scale, shape=shape):
            meaneps = mean - epsilon
            parameter_guess = [0.1]
            par = opt.fsolve(
                self.weibull_parameter,
                parameter_guess,
                args=(meaneps, std),
            )
            k = par[0]
            u_1 = meaneps / (spec.gamma(1 + 1 / k)) + epsilon
            scale = u_1 - epsilon
        else:
            k = shape

        # use scipy to do the heavy lifting
        self.dist_obj = weibull(c=k, loc=epsilon, scale=scale)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Weibull"

    def weibull_parameter(
        self, x: float | np.ndarray, *args: float
    ) -> float | np.ndarray:
        """Return the moment residual for Weibull shape fitting.

        The optimizer supplies the trial shape x; args contains mean minus
        the lower bound and the standard deviation.
        """
        meaneps, std = args
        f = (spec.gamma(1 + 2 / x) - (spec.gamma(1 + 1 / x)) ** 2) ** 0.5 - (
            std / meaneps
        ) * spec.gamma(1 + 1 / x)
        return f
