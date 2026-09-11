from scipy.special import erf
import numpy as np
from scipy.stats import lognorm
from .distribution import Distribution
from .lognormal import Lognormal

__all__ = ["ShiftedLognormal"]


class ShiftedLognormal(Lognormal):
    """Shifted Lognormal distribution

    If X is a lognormal random variable, then Y = X + lower is a shifted lognormal random variable.

    :Arguments:
      - name (str):         Name of the random variable
      - mean (float):       Mean
      - std (float):       Standard deviation\n
      - lower (float):      Lower bound of the distribution (i.e. the shift applied to the lognormal)\n
      - start_point (float): Start point for seach\n

    Note: Could use scipy to do the heavy lifting. However, there is a small
    performance hit, so for this common dist use bespoke implementation
    for the PDF, CDF.
    """

    def __init__(self, name, mean, std, lower, *, start_point=None):
        self._ctor_kwargs = {"lower": lower}

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

    def pdf(self, x):
        """
        Probability density function
        Note: asssumes x>lower for performance, scipy manages this appropriately
        """
        return super().pdf(x - self.lower)

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        return super().cdf(x - self.lower)

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        return super().u_to_x(u) + self.lower

    def x_to_u(self, x):
        """
        Transformation from x to u
        Note: asssumes x>lower for performance
        """
        return super().x_to_u(x - self.lower)

    def set_lower(self, lower=0):
        """
        Updating the distribution lower parameter.
        For Lognormal, even though we have a SciPy object, it's not being used in the
        functions above for performance, so we need to update params directly.
        """
        self._update_params(self.mean, self.std, lower)
        self.lower = lower
