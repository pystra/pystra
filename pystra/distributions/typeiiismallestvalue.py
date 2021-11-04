#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from .distribution import Distribution

from .weibull import Weibull


class TypeIIIsmallestValue(Distribution):
    """Type III smallest value distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean or u_1\n
      - stdv (float): Standard deviation or k\n
      - epsilon (float): Epsilon\n
      - input_type (any): Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, epsilon=0, input_type=None, startpoint=None):

        # This distribution is the same as the Weibull - keep for backwards compat
        dist = Weibull(name, mean, stdv, epsilon, input_type, startpoint)
        self.dist_obj = dist.dist_obj

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "TypeIIIsmallestValue"
