#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats import rv_continuous

from .distribution import Distribution
from ..errors import ModelError

__all__ = ["ScipyDist"]


class ScipyDist(Distribution):
    """Distribution wrapper for a frozen Scipy Stats Distribution object

    Discrete random variables not yet supported.

    :Attributes:
      - name (str):             Name of the random variable\n
      - dist_obj (Scipy dist):  The Scipy distribution object\n
      - start_point (float):     Start point for seach\n
    """

    _native_parameters = ()

    def _parameter_values(self):
        return {"dist_obj": self.dist_obj}

    @property
    def sensitivity_params(self):
        """No generic moment perturbation; replace the constructor inputs."""
        return {}

    def __init__(self, name, dist_obj, start_point=None):
        if not isinstance(dist_obj, rv_frozen):
            raise ModelError(
                f"ScipyDist {name} requires a frozen Scipy distribution object"
            )
        if not isinstance(dist_obj.dist, rv_continuous):
            raise ModelError(f"ScipyDist {name} requires a continuous distribution")

        super().__init__(
            name=name,
            dist_obj=dist_obj,
            start_point=start_point,
        )

        self.dist_type = "ScipyDist"
