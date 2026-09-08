import numpy as np
from scipy.stats import genextreme
from scipy.special import gamma
from pystra.distributions import Distribution

__all__ = ["GEVmax", "GEVmin"]


class GEVmax(Distribution):
    """Generalized Extreme Value (GEV) distribution for maxima.

    This distribution unifies the different types of extreme value
    distributions: Gumbel (Type I), Fréchet (Type II), and
    Weibull (Type III).

    :Arguments:
        - name (str):       Name of the random variable\n
        - mean (float):     Mean\n
        - stdv (float):     Standard deviation\n
        - shape (float):    Shape parameter. shape < 0.0 is Weibull,
          shape > 0 is Frechet.\n
        - input_type (any): Change meaning of mean and stdv\n
        - startpoint (float): Start point for seach\n

    :Raises:
        - ValueError: If `shape` is greater than or equal to 0.5

    :Notes:
        - The shape parameter `shape` must be less than 0.5 for
          finite variance.
        - `shape` < 0 is the Weibull case, `shape` = 0 is the
          Gumbel case, and `shape` > 0 is the Fréchet case.
        - This distribution is to model maxima.
    """

    def __init__(self, name, mean, stdv, shape, input_type=None, startpoint=None):
        if shape >= 0.5:
            raise ValueError("`shape` must be less than 0.5 for finite variance")

        self.shape = shape
        self._ctor_kwargs = {"shape": shape}

        g1 = gamma(1 - shape)
        g2 = gamma(1 - 2 * shape)

        if input_type is None:
            if np.isclose(shape, 0):
                scale = stdv * np.sqrt(6) / np.pi
                loc = mean - scale * np.euler_gamma
            else:
                scale = stdv * np.abs(shape) / np.sqrt(g2 - g1**2)
                loc = mean - scale / shape * (g1 - 1)
        else:
            loc = mean
            scale = stdv
            if np.isclose(shape, 0):
                self.mean = loc + scale * np.euler_gamma
                self.stdv = scale * np.pi / np.sqrt(6)
            else:
                self.mean = loc + (g1 - 1) * scale / shape
                self.stdv = np.sqrt((g2 - g1**2) * (scale / shape) ** 2)

        # use scipy to do the heavy lifting
        self.dist_obj = genextreme(c=-shape, loc=loc, scale=scale)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "GEVmax"

    @property
    def sensitivity_params(self):
        r"""Sensitivity parameters for GEVmax.

        Returns ``{"mean": μ, "std": σ, "shape": ξ}``.  The shape
        parameter ξ controls tail behaviour: ξ < 0 is Weibull (bounded
        upper tail), ξ = 0 is Gumbel, ξ > 0 is Fréchet (heavy-tailed).
        """
        return {"mean": self.mean, "std": self.stdv, "shape": self.shape}


class GEVmin(Distribution):
    """Generalized Extreme Value (GEV) distribution for minima.

    This distribution unifies the different types of extreme value distributions: Gumbel (Type I), Fréchet (Type II), and Weibull (Type III).

    :Arguments:
        - name (str):       Name of the random variable\n
        - mean (float):     Mean\n
        - stdv (float):     Standard deviation\n
        - shape (float):       Shape parameter. shape < 0.0 is Weibull, shape > 0 is Frechet.\n
        - input_type (any): Change meaning of mean and stdv\n
        - startpoint (float): Start point for seach\n

    :Raises:
        - ValueError: If `shape` is greater than or equal to 0.5

    :Notes:
        - The shape parameter `shape` must be less than 0.5 for finite variance.
        - `shape` < 0 is the Weibull case, `shape` = 0 is the Gumbel case, and `shape` > 0 is the Fréchet case.
        - This distribution is to model minima.
    """

    def __init__(self, name, mean, stdv, shape, input_type=None, startpoint=None):
        if shape >= 0.5:
            raise ValueError("`shape` must be less than 0.5 for finite variance")

        self.shape = shape
        self._ctor_kwargs = {"shape": shape}

        g1 = gamma(1 - shape)
        g2 = gamma(1 - 2 * shape)

        if input_type is None:
            # mean and stdv passed in
            self.mean = mean
            self.stdv = stdv
            if np.isclose(shape, 0):
                scale = self.stdv * np.sqrt(6) / np.pi
                loc = self.mean - scale * np.euler_gamma
            else:
                scale = self.stdv * np.abs(shape) / np.sqrt(g2 - g1**2)
                loc = self.mean - (scale / shape) * (g1 - 1)
        else:
            # loc and scale are actual GEV parameters
            loc = mean
            scale = stdv
            if np.isclose(shape, 0):
                self.mean = loc + scale * np.euler_gamma
                self.stdv = scale * np.pi / np.sqrt(6)
            else:
                self.mean = loc + (g1 - 1) * scale / shape
                self.stdv = np.sqrt((g2 - g1**2) * (scale / shape) ** 2)

        # use scipy to do the heavy lifting; note reverse shape sign convention
        self.dist_obj = genextreme(c=-shape, loc=-loc, scale=scale)

        # Save correct moments before super().__init__() — the dist_obj
        # represents -X internally, so dist_obj.mean() gives the wrong
        # sign and would overwrite the correct values via _update_moments()
        _correct_mean = self.mean
        _correct_stdv = self.stdv

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        # Restore correct moments (dist_obj models -X internally)
        self.mean = _correct_mean
        self.stdv = _correct_stdv
        if startpoint is None:
            self.startpoint = self.mean

        self.dist_type = "GEVmin"

    @property
    def sensitivity_params(self):
        r"""Sensitivity parameters for GEVmin.

        Returns ``{"mean": μ, "std": σ, "shape": ξ}``.  The shape
        parameter ξ controls tail behaviour: ξ < 0 is Weibull (bounded
        lower tail), ξ = 0 is Gumbel, ξ > 0 is Fréchet (heavy-tailed).
        """
        return {"mean": self.mean, "std": self.stdv, "shape": self.shape}

    def pdf(self, x):
        """
        Probability density function
        """
        return self.dist_obj.pdf(-x)

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        return 1 - self.dist_obj.cdf(-x)

    def ppf(self, u):
        """
        Inverse cumulative distribution function
        """
        x = self.dist_obj.ppf(u)
        return -x
