#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import expon

from .distribution import Distribution


class ShiftedExponential(Distribution):
    """Shifted exponential distribution

    :Attributes:
        - name (str):         Name of the random variable\n
        - mean (float):       Mean or lamb\n
        - stdv (float):       Standard deviation or x_zero\n
        - input_type (any):   Change meaning of mean and stdv\n
        - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):

        if input_type is None:
            x_zero = mean - stdv
            lamb = 1 / stdv
        else:
            lamb = mean
            x_zero = stdv

        # use scipy to do the heavy lifting
        self.dist_obj = expon(loc=x_zero, scale=1 / lamb)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "ShiftedExponential"
