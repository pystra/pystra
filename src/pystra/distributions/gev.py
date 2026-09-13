import numpy as np
from scipy.stats import genextreme
from scipy.special import gamma
from pystra.distributions import Distribution
from .distribution import _uses_native_parameters

__all__ = ["GEV", "GEVmax", "GEVMin"]


class GEV(Distribution):
    """Generalized Extreme Value (GEV) distribution for maxima.

    ``GEVmax`` is an alias for this class.

    This distribution unifies the different types of extreme value
    distributions: Gumbel (Type I), Fréchet (Type II), and
    Weibull (Type III).

    :Arguments:
        - name (str):       Name of the random variable\n
        - mean (float):     Mean\n
        - std (float):     Standard deviation\n
        - shape (float):    Shape parameter. shape < 0.0 is Weibull,
          shape > 0 is Frechet.\n
        - loc (float): Location, given instead of mean and std\n
        - scale (float): Scale, given instead of mean and std\n
        - start_point (float): Start point for seach\n

    :Raises:
        - ValueError: If `shape` is greater than or equal to 0.5

    :Notes:
        - The shape parameter `shape` must be less than 0.5 for
          finite variance.
        - `shape` < 0 is the Weibull case, `shape` = 0 is the
          Gumbel case, and `shape` > 0 is the Fréchet case.
        - This distribution is to model maxima.
    """

    def __init__(
        self,
        name,
        mean=None,
        std=None,
        shape=None,
        *,
        loc=None,
        scale=None,
        start_point=None,
    ):
        if shape is None:
            raise TypeError(f"{type(self).__name__} needs shape")
        if shape >= 0.5:
            raise ValueError("`shape` must be less than 0.5 for finite variance")

        self.shape = shape
        self._ctor_kwargs = {"shape": shape}

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
    def sensitivity_params(self):
        r"""Sensitivity parameters for GEV.

        Returns ``{"mean": μ, "std": σ, "shape": ξ}``.  The shape
        parameter ξ controls tail behaviour: ξ < 0 is Weibull (bounded
        upper tail), ξ = 0 is Gumbel, ξ > 0 is Fréchet (heavy-tailed).
        """
        return {"mean": self.mean, "std": self.std, "shape": self.shape}


# ``GEVmax`` is kept as another name for ``GEV``: the one deliberate alias in
# the 2.0 API, at the maintainer's request.
GEVmax = GEV


class GEVMin(Distribution):
    """Generalized Extreme Value (GEV) distribution for minima.

    This distribution unifies the different types of extreme value distributions: Gumbel (Type I), Fréchet (Type II), and Weibull (Type III).

    :Arguments:
        - name (str):       Name of the random variable\n
        - mean (float):     Mean\n
        - std (float):     Standard deviation\n
        - shape (float):       Shape parameter. shape < 0.0 is Weibull, shape > 0 is Frechet.\n
        - loc (float): Location, given instead of mean and std\n
        - scale (float): Scale, given instead of mean and std\n
        - start_point (float): Start point for seach\n

    :Raises:
        - ValueError: If `shape` is greater than or equal to 0.5

    :Notes:
        - The shape parameter `shape` must be less than 0.5 for finite variance.
        - `shape` < 0 is the Weibull case, `shape` = 0 is the Gumbel case, and `shape` > 0 is the Fréchet case.
        - This distribution is to model minima.
    """

    def __init__(
        self,
        name,
        mean=None,
        std=None,
        shape=None,
        *,
        loc=None,
        scale=None,
        start_point=None,
    ):
        if shape is None:
            raise TypeError(f"{type(self).__name__} needs shape")
        if shape >= 0.5:
            raise ValueError("`shape` must be less than 0.5 for finite variance")

        self.shape = shape
        self._ctor_kwargs = {"shape": shape}

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
    def sensitivity_params(self):
        r"""Sensitivity parameters for GEVMin.

        Returns ``{"mean": μ, "std": σ, "shape": ξ}``.  The shape
        parameter ξ controls tail behaviour: ξ < 0 is Weibull (bounded
        lower tail), ξ = 0 is Gumbel, ξ > 0 is Fréchet (heavy-tailed).
        """
        return {"mean": self.mean, "std": self.std, "shape": self.shape}

    def pdf(self, x):
        """
        Probability density function
        """
        return self.dist_obj.pdf(-np.asarray(x, dtype=float))

    def logpdf(self, x):
        """Log density."""
        return self.dist_obj.logpdf(-np.asarray(x, dtype=float))

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        return self.dist_obj.sf(-np.asarray(x, dtype=float))

    def sf(self, x):
        """Survival function."""
        return self.dist_obj.cdf(-np.asarray(x, dtype=float))

    def _lower_logcdf(self, x):
        return self.dist_obj.logsf(-np.asarray(x, dtype=float))

    def _upper_logsf(self, x):
        return self.dist_obj.logcdf(-np.asarray(x, dtype=float))

    def ppf(self, u):
        """
        Inverse cumulative distribution function
        """
        return -self.dist_obj.isf(u)

    def isf(self, q):
        """Inverse survival function."""
        return -self.dist_obj.ppf(q)
