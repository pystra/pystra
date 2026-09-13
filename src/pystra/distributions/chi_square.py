"""Chi square marginal distribution."""

from scipy.stats import chi2

from .distribution import Distribution, _uses_native_parameters

__all__ = ["ChiSquare"]


class ChiSquare(Distribution):
    """Chi-square distribution.

    Supply either mean and std or ``df``.
    The two parameterizations cannot be combined.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    mean : float, optional
        Mean in physical space.
    std : float, optional
        Standard deviation in physical space.
    df : float, optional
        Degrees of freedom. Supply instead of mean and std.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.

    Notes
    -----
    The moment parameterization should satisfy mean = std**2 / 2.
    Use df to specify the chi-square law directly.
    """

    _native_parameters = ("df",)

    def _parameter_values(self):
        return {"df": self.nu}

    def __init__(self, name, mean=None, std=None, *, df=None, start_point=None):
        if not _uses_native_parameters(self, mean, std, df=df):
            lamb = 0.5
            mean_test = lamb * std**2
            if mean / mean_test < 0.95 or mean / mean_test > 1.05:
                print(
                    "Error when using Chi-square distribution. "
                    "Mean and std should be given such that mean = 0.5*std.**2\n"
                )
            nu = 2 * (mean**2) / (std**2)
        else:
            nu = df

        self.nu = nu

        # use scipy to do the heavy lifting
        self.dist_obj = chi2(df=nu)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "ChiSquare"
