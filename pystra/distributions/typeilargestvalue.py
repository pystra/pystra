#!/usr/bin/python -tt
# -*- coding: utf-8 -*-


from .distribution import Distribution
from .gumbel import Gumbel


class TypeIlargestValue(Distribution):
    """Type I largest value distribution

    :Attributes:
      - name (str):   Name of the random variable\n
      - mean (float): Mean or u_n\n
      - stdv (float): Standard deviation or a_n\n
      - input_type (any): Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):

        # This distribution is the same as the Gumbel - keep for backwards compat
        dist = Gumbel(name, mean, stdv, input_type, startpoint)
        self.dist_obj = dist.dist_obj

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "TypeIlargestValue"
