#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt


class StdNormal:
    """
    A performant implementation of the standard normal distribution providing
    the basic functions PDF, CDF, and inverse CDF, since these are used
    frequently in the algorithms.
    """

    @staticmethod
    def pdf(u):
        p = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        return p

    @staticmethod
    def cdf(u):
        p = 0.5 + sp.erf(u / np.sqrt(2)) / 2
        return p

    @staticmethod
    def inv_cdf(p):
        u = sp.erfinv(2 * p - 1) * np.sqrt(2)
        return u


class Constant:
    """
    A deterministic variable in the limit state function
    """

    def __init__(self, name, val):
        self.name = name
        self.val = val

    def getName(self):
        return self.name

    def getValue(self):
        return self.val


class Distribution:
    """Probability distribution

    Attributes:
      name (str):   Name of the random variable\n
      dist_obj (SciPy rv): if subclassing SciPy distribution
      mean (float): Mean or other variable\n
      stdv (float): Standard deviation or other variable\n
      startpoint (float): Start point for seach\n
      p1 (float):   Parameter for the distribution\n
      p2 (float):   Parameter for the distribution\n
      p3 (float):   Parameter for the distribution\n
      p4 (float):   Parameter for the distribution\n
      input_type (any): Change meaning of mean and stdv\n

      Default: all values
    """

    std_normal = StdNormal()

    def __init__(
        self, name="", dist_obj=None, mean=None, stdv=None, startpoint=None,
    ):
        self.name = name
        self.dist_type = "BaseCls"

        # This is the key object that is to be defined in derived classes that
        # are using the base class functionality
        self.dist_obj = dist_obj

        if self.dist_obj is not None:
            self.mean = self.dist_obj.mean()
            self.stdv = self.dist_obj.std()
        elif mean is None or stdv is None:
            raise Exception("Mean and std dev must be defined in derived classes")
        else:
            self.mean = mean
            self.stdv = stdv

        self.setStartPoint(startpoint)

    def __repr__(self):
        string = self.name + ": " + self.dist_type + " distribution"
        return string

    def getName(self):
        return self.name

    def getMean(self):
        return self.mean

    def getStdv(self):
        return self.stdv

    def getStartPoint(self):
        return self.startpoint

    def setStartPoint(self, startpoint=None):
        if startpoint is None:
            self.startpoint = self.mean
        else:
            self.startpoint = startpoint

    # The following can be overridden by derived classes to implement more
    # efficient calculations where desirable

    def pdf(self, x):
        """
        Probability density function
        """
        return self.dist_obj.pdf(x)

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        return self.dist_obj.cdf(x)

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        return self.dist_obj.ppf(self.std_normal.cdf(u))

    def x_to_u(self, x):
        """
        Transformation from x to u
        """
        u = self.std_normal.inv_cdf(self.cdf(x))
        return u

    def jacobian(self, u, x):
        """
        Compute the Jacobian
        """
        pdf1 = self.pdf(x)
        pdf2 = self.std_normal.pdf(u)
        J = np.diag(pdf1 / pdf2)
        return J

    def plot(self, ax=None):
        """
        Plots the PDF of the distribution
        """
        # auto-range
        n = 1000
        u = np.random.rand(n)
        samples = self.dist_obj.ppf(u)
        x = np.linspace(np.min(samples), np.max(samples), 100)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, self.pdf(x))
        ax.set_title(self.name)
