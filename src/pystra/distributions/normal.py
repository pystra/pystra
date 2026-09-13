"""Normal marginal distribution."""

import numpy as np
from scipy import special as sp

from .distribution import Distribution

__all__ = ["Normal"]


class Normal(Distribution):
    """Normal distribution using direct standard normal formulas.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float
        Mean in physical space.
    std : float
        Standard deviation in physical space.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    def __init__(self, name, mean, std, *, start_point=None):
        """
        Leave initialization to the base class
        """
        super().__init__(
            name=name,
            mean=mean,
            std=std,
            start_point=start_point,
        )
        self.dist_type = "Normal"

    def pdf(self, x):
        """
        probability density function
        """
        z = (x - self.mean) / self.std
        p = self.std_normal.pdf(z) / self.std
        return p

    def cdf(self, x):
        """
        cumulative distribution function
        """
        z = (x - self.mean) / self.std
        p = self.std_normal.cdf(z)
        return p

    def ppf(self, p):
        """
        inverse cumulative distribution function
        """
        z = self.std_normal.ppf(p)
        x = self.std * z + self.mean
        return x

    def sf(self, x):
        """Survival function."""
        return sp.ndtr((self.mean - x) / self.std)

    def isf(self, q):
        """Inverse survival function."""
        return self.mean - self.std * sp.ndtri(q)

    def logpdf(self, x):
        """Log density."""
        z = (x - self.mean) / self.std
        return -0.5 * z**2 - np.log(self.std) - 0.5 * np.log(2 * np.pi)

    def logcdf(self, x):
        """Log CDF."""
        return sp.log_ndtr((x - self.mean) / self.std)

    def logsf(self, x):
        """Log survival function."""
        return sp.log_ndtr((self.mean - x) / self.std)

    def _lower_quantile_log(self, logp):
        return self.mean + self.std * sp.ndtri_exp(logp)

    def _upper_quantile_log(self, logq):
        return self.mean - self.std * sp.ndtri_exp(logq)

    def sample(self, n=1000):
        """
        Override sample from base class due to bespoke implementation
        """
        u = np.random.rand(n)
        samples = self.ppf(u)
        return samples

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        x = u * self.std + self.mean
        return x

    def x_to_u(self, x):
        """
        Transformation from x to u
        """
        u = (x - self.mean) / self.std
        return u

    def jacobian(self, u, x):
        """
        Compute the Jacobian  (e.g. Lemaire, eq. 4.9)
        For the Normal distribution, the more usual general function can be
        specialized as follows.
        """
        J = np.diag(np.repeat(1 / self.std, u.size))
        return J

    def cdf_gradient(self, x):
        r"""Analytical derivatives of the Normal CDF w.r.t. μ and σ.

        .. math::
            \frac{\partial F}{\partial \mu} = -\frac{1}{\sigma}\,
                \varphi\!\left(\frac{x - \mu}{\sigma}\right)

            \frac{\partial F}{\partial \sigma} = -\frac{x - \mu}{\sigma^2}\,
                \varphi\!\left(\frac{x - \mu}{\sigma}\right)
        """
        z = (x - self.mean) / self.std
        phi_z = self.std_normal.pdf(z)
        dF_dmu = -phi_z / self.std
        dF_dsig = -phi_z * z / self.std
        return {"mean": dF_dmu, "std": dF_dsig}

    def set_location(self, loc=0):
        """
        Updating the distribution location parameter. For Normal, there is no need to
        update other properties as a result of this change.
        """
        self._mean = loc

    def set_scale(self, scale=1):
        """
        Updating the distribution scale parameter. For Normal, there is no need to
        update other properties as a result of this change.
        """
        self._std = scale
