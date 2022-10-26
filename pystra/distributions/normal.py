#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

from .distribution import Distribution


class Normal(Distribution):
    """Normal distribution

    :Attributes:
      - name (str):         Name of the random variable\n
      - mean (float):       Mean\n
      - stdv (float):       Standard deviation\n
      - input_type (any):   Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n

    Note: while we could use SciPy norm distribution here, there is a
    substantial perfromance hit, so use local implementation.
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):
        """
        Leave initialization to the base class
        """
        super().__init__(
            name=name,
            mean=mean,
            stdv=stdv,
            startpoint=startpoint,
        )
        self.dist_type = "Normal"

    def pdf(self, x):
        """
        probability density function
        """
        z = (x - self.mean) / self.stdv
        p = self.std_normal.pdf(z) / self.stdv
        return p

    def cdf(self, x):
        """
        cumulative distribution function
        """
        z = (x - self.mean) / self.stdv
        p = self.std_normal.cdf(z)
        return p

    def ppf(self, p):
        """
        inverse cumulative distribution function
        """
        z = self.std_normal.ppf(p)
        x = self.stdv * z + self.mean
        return x

    def sample(self, n=1000):
        """
        Override sample from base class due to bespoke implementation
        """
        u = np.random.rand(n)
        samples = self.ppf(u)
        return samples

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        x = u * self.stdv + self.mean
        return x

    def x_to_u(self, x):
        """
        Transformation from x to u
        """
        u = (x - self.mean) / self.stdv
        return u

    def jacobian(self, u, x):
        """
        Compute the Jacobian  (e.g. Lemaire, eq. 4.9)
        For the Normal distribution, the more usual general function can be
        specialized as follows.
        """
        J = np.diag(np.repeat(1 / self.stdv, u.size))
        return J

    def set_location(self, loc=0):
        """
        Updating the distribution location parameter. For Normal, there is no need to
        update other properties as a result of this change.
        """
        self.mean = loc

    def set_scale(self, scale=1):
        """
        Updating the distribution scale parameter. For Normal, there is no need to
        update other properties as a result of this change.
        """
        self.stdv = scale
