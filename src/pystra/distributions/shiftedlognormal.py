from scipy.special import erf
import numpy as np
from scipy.stats import lognorm
from .distribution import Distribution

class ShiftedLognormal(Distribution):
    """Shifted Lognormal distribution

    If X is a lognormal random variable, then Y = X + lower is a shifted lognormal random variable.

    :Arguments:
      - name (str):         Name of the random variable
      - mean (float):       Mean
      - stdv (float):       Standard deviation\n
      - lower (float):      Lower bound of the distribution (i.e. the shift applied to the lognormal)\n
      - input_type (any):   Change meaning of mean and stdv. Not implemented!\n
      - startpoint (float): Start point for seach\n

    Note: Could use scipy to do the heavy lifting. However, there is a small
    performance hit, so for this common dist use bespoke implementation
    for the PDF, CDF.
    """
    def __init__(self, name, mean, stdv, lower, input_type=None, startpoint=None):
        if input_type is not None:
            raise NotImplementedError("`input_type` not implemented")

        self.mean = mean
        self.stdv = stdv
        self._update_params(mean, stdv, lower)

        self.dist_obj = lognorm(scale=np.exp(self.lamb), s=self.zeta, loc=self.lower)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "ShiftedLognormal"

    def _update_params(self, mean, stdv, lower):
        shifted_mean = mean - lower
        cov = stdv / shifted_mean
        self.zeta = (np.log(1 + cov**2)) ** 0.5
        self.lamb = np.log(shifted_mean) - 0.5 * self.zeta**2
        self.lower = lower

    def pdf(self, x):
        """
        Probability density function
        Note: asssumes x>lower for performance, scipy manages this appropriately
        """
        t = x - self.lower
        z = (np.log(t) - self.lamb) / self.zeta
        p = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * self.zeta * t)
        return p  # self.lognormal.pdf(t)
    
    def cdf(self, x):
        """
        Cumulative distribution function
        """
        t = x - self.lower
        z = (np.log(t) - self.lamb) / self.zeta
        p = 0.5 + erf(z / np.sqrt(2)) / 2
        return p  # self.lognormal.cdf(t)
    
    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        t = np.exp(u * self.zeta + self.lamb)
        return t + self.lower
    
    def x_to_u(self, x):
        """
        Transformation from x to u
        Note: asssumes x>lower for performance
        """
        t = x - self.lower
        u = (np.log(t) - self.lamb) / self.zeta
        return u
    
    def set_location(self, loc=0):
        """
        Updating the distribution location parameter.
        For Lognormal, even though we have a SciPy object, it's not being used in the
        functions above for performance, so we need to update pe.arams directly.
        """

        self._update_params(loc, self.stdv, self.lower)
        self.mean = loc

    def set_scale(self, scale=1):
        """
        Updating the distribution scale parameter.
        For Lognormal, even though we have a SciPy object, it's not being used in the
        functions above for performance, so we need to update params directly.
        """
        self._update_params(self.mean, scale, self.lower)
        self.stdv = scale

    def set_lower(self, lower=0):
        """
        Updating the distribution lower parameter.
        For Lognormal, even though we have a SciPy object, it's not being used in the
        functions above for performance, so we need to update params directly.
        """
        self._update_params(self.mean, self.stdv, lower)
        self.lower = lower
