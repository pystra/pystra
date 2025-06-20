import numpy as np
from scipy.stats import genextreme
from scipy.special import gamma
from pystra.distributions import Distribution

class Genextreme(Distribution):
    """Generalized Extreme Value (GEV) distribution for maxima.

    This distribution the different types of extreme value distributions: Gumbel (Type I), Fréchet (Type II), and Weibull (Type III).

    :Arguments:
        - name (str):       Name of the random variable\n
        - mean (float):     Mean\n
        - stdv (float):     Standard deviation\n
        - xi (float):       Shape parameter. xi < 0.0 is Weibull, xi > 0 is Frechet.\n
        - input_type (any): Change meaning of mean and stdv\n
        - startpoint (float): Start point for seach\n

    :Raises:
        - ValueError: If `xi` is greater than or equal to 0.5

    :Notes:
        - The shape parameter `xi` must be less than 0.5 for finite variance.
        - `xi` < 0 is the Weibull case, `xi` = 0 is the Gumbel case, and `xi` > 0 is the Fréchet case.
        - This distribution is to model maxima.
    """

    def __init__(self, name, mean, stdv, xi, input_type=None, startpoint=None):
        if input_type is not None:
            raise NotImplementedError("`input_type` not implemented")
        
        if xi >= 0.5:
            raise ValueError("`xi` must be less than 0.5 for finite variance")
        elif np.isclose(xi, 0):
            sigma = stdv * np.sqrt(6) / np.pi
            mu = mean - sigma * np.euler_gamma
        else:
            g1 = gamma(1-xi)
            g2 = gamma(1-2*xi)
            sigma = stdv * np.abs(xi) / np.sqrt(g2 - g1**2)
            mu = mean - sigma/xi * (g1-1)

        # use scipy to do the heavy lifting
        self.dist_obj = genextreme(c=-xi, loc=mu, scale=sigma)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "GEV"