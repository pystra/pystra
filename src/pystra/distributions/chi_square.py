#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import chi2

from .distribution import Distribution, _uses_native_parameters

__all__ = ["ChiSquare"]


class ChiSquare(Distribution):
    """Chi-Square distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean\n
      - std (float): Standard deviation\n
      - df (float): Degrees of freedom, given instead of mean and std\n
      - start_point (float): Start point for seach\n
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
