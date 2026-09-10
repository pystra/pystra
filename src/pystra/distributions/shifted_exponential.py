#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from scipy.stats import expon

from .distribution import Distribution

__all__ = ["ShiftedExponential"]


class ShiftedExponential(Distribution):
    """Shifted exponential distribution

    :Attributes:
        - name (str):         Name of the random variable\n
        - mean (float):       Mean or lamb\n
        - std (float):       Standard deviation or x_zero\n
        - input_type (any):   Change meaning of mean and std\n
        - start_point (float): Start point for seach\n
    """

    def __init__(self, name, mean, std, input_type=None, start_point=None):
        if input_type is None:
            x_zero = mean - std
            lamb = 1 / std
        else:
            lamb = mean
            x_zero = std

        # use scipy to do the heavy lifting
        self.dist_obj = expon(loc=x_zero, scale=1 / lamb)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            start_point=start_point,
        )

        self.dist_type = "ShiftedExponential"
