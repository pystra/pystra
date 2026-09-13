"""Gamma marginal distribution."""

from scipy.stats import gamma

from .distribution import Distribution, _uses_native_parameters

__all__ = ["Gamma"]


class Gamma(Distribution):
    """Gamma distribution.

    Supply either mean and std or ``shape`` and ``rate``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    rate : float, optional
        Positive rate parameter (the reciprocal of scale). Supply with the other native parameters instead of mean and std.
    shape : float, optional
        Shape parameter. Supply with the other native parameters instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ("shape", "rate")

    def _parameter_values(self):
        return {
            "shape": self.dist_obj.kwds["a"],
            "rate": 1 / self.dist_obj.kwds["scale"],
        }

    def __init__(
        self, name, mean=None, std=None, *, rate=None, shape=None, start_point=None
    ):
        if _uses_native_parameters(self, mean, std, rate=rate, shape=shape):
            beta = rate
            alpha = shape
        else:
            beta = mean / (std**2)
            alpha = mean**2 / (std**2)

        # use scipy to do the heavy lifting
        self.dist_obj = gamma(a=alpha, scale=1 / beta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "Gamma"
