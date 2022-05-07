#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import chi2

from .distribution import Distribution


class ChiSquare(Distribution):
    """Chi-Square distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean or nu\n
      - stdv (float): Standard deviation\n
      - input_type (any): Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv=None, input_type=None, startpoint=None):

        if input_type is None:
            lamb = 0.5
            mean_test = lamb * stdv**2
            if mean / mean_test < 0.95 or mean / mean_test > 1.05:
                print(
                    "Error when using Chi-square distribution. "
                    "Mean and stdv should be given such that mean = 0.5*stdv.**2\n"
                )
            nu = 2 * (mean**2) / (stdv**2)
        else:
            nu = mean

        self.nu = nu

        # use scipy to do the heavy lifting
        self.dist_obj = chi2(df=nu)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "ChiSquare"
