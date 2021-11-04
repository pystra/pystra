#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats import rv_continuous

from .distribution import Distribution


class ScipyDist(Distribution):
    """Distribution wrapper for a frozen Scipy Stats Distribution object

    Discrete random variables not yet supported.

    :Attributes:
      - name (str):             Name of the random variable\n
      - dist_obj (Scipy dist):  The Scipy distribution object\n
      - startpoint (float):     Start point for seach\n
    """

    def __init__(self, name, dist_obj, startpoint=None):

        if not isinstance(dist_obj, rv_frozen):
            raise Exception(
                f"ScipyDist {name} requires a frozen Scipy distribution object"
            )
        if not isinstance(dist_obj.dist, rv_continuous):
            raise Exception(f"ScipyDist {name} requires a continuous distribution")

        super().__init__(
            name=name,
            dist_obj=dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "ScipyDist"
