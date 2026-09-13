import numpy as np
from scipy.stats import lognorm
from .distribution import Distribution
from .lognormal import Lognormal

__all__ = ["ShiftedLognormal"]


class ShiftedLognormal(Lognormal):
    """Shifted lognormal distribution.

    If X is lognormal, Y = X + lower has this distribution. The supplied
    mean and standard deviation describe Y.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float
        Mean in physical space.
    std : float
        Standard deviation in physical space.
    lower : float
        Lower bound of the distribution.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ()

    def _parameter_values(self):
        return {"mean": self.mean, "std": self.std, "lower": self.lower}

    def __init__(self, name, mean, std, lower, *, start_point=None):

        self._mean = mean
        self._std = std
        self.lower = None
        self._update_params(mean, std, lower)

        self.dist_obj = lognorm(scale=np.exp(self.lamb), s=self.zeta, loc=self.lower)

        Distribution.__init__(
            self,
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "ShiftedLognormal"

    def _update_params(self, mean, std, lower=None):
        lower = self.lower if lower is None else lower
        super()._update_params(mean - lower, std)
        self.lower = lower

    @property
    def _shift(self):
        return self.lower

    def set_lower(self, lower=0):
        """
        Updating the distribution lower parameter.
        For Lognormal, even though we have a SciPy object, it's not being used in the
        functions above for performance, so we need to update params directly.
        """
        self._update_params(self.mean, self.std, lower)
        self.lower = lower
