#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import gamma

from .distribution import Distribution


class Gamma(Distribution):
    """Gamma distribution

    :Attributes:
      - name (str):         Name of the random variable\n
      - mean (float):       Mean or beta\n
      - stdv (float):       Standard deviation or k\n
      - input_type (any):   Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):

        if input_type is None:
            beta = mean / (stdv**2)
            alpha = mean**2 / (stdv**2)
        else:
            beta = mean
            alpha = stdv

        # use scipy to do the heavy lifting
        self.dist_obj = gamma(a=alpha, scale=1 / beta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "Gamma"
