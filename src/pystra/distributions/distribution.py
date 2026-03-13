#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt
from scipy.stats._distn_infrastructure import rv_frozen


class StdNormal:
    """Standard normal distribution (mean 0, standard deviation 1).

    A lightweight implementation using ``scipy.special`` error functions,
    avoiding the overhead of a full ``scipy.stats`` distribution object.
    This class is used internally by the Nataf transformation and by the
    marginal distribution mappings (``x_to_u`` / ``u_to_x``).
    """

    @staticmethod
    def pdf(u):
        """Probability density function of the standard normal.

        Parameters
        ----------
        u : float or array_like
            Quantile(s) in standard normal space.

        Returns
        -------
        float or ndarray
            Density value(s).
        """
        p = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        return p

    @staticmethod
    def cdf(u):
        """Cumulative distribution function of the standard normal.

        Parameters
        ----------
        u : float or array_like
            Quantile(s) in standard normal space.

        Returns
        -------
        float or ndarray
            Probability value(s) in [0, 1].
        """
        p = 0.5 + sp.erf(u / np.sqrt(2)) / 2
        return p

    @staticmethod
    def ppf(p):
        """Percent-point (inverse CDF) of the standard normal.

        Parameters
        ----------
        p : float or array_like
            Probability value(s) in (0, 1).

        Returns
        -------
        float or ndarray
            Quantile(s) in standard normal space.
        """
        u = sp.erfinv(2 * p - 1) * np.sqrt(2)
        return u


class Constant:
    """A deterministic (non-random) variable in the limit state function.

    Constants are included in the stochastic model but are not treated as
    random variables — they carry a fixed value through every evaluation of
    the limit state function.

    Parameters
    ----------
    name : str
        Name of the constant (must match a keyword argument of the limit
        state function).
    val : float
        The fixed value.
    """

    def __init__(self, name, val):
        self.name = name
        self.val = val

    def getName(self):
        """Return the constant name."""
        return self.name

    def getValue(self):
        """Return the constant value."""
        return self.val


class Distribution:
    r"""Base class for all probability distributions used in reliability analysis.

    Subclasses typically construct a ``scipy.stats`` frozen distribution
    object (``dist_obj``) and pass it to this base class, which then
    delegates ``pdf``, ``cdf``, ``ppf``, and the Nataf-space
    transformations (``x_to_u``, ``u_to_x``, ``jacobian``) to it.

    Subclasses that do not wrap a SciPy distribution must override the
    transformation and Jacobian methods directly (see, e.g.,
    :class:`ZeroInflated`).

    Parameters
    ----------
    name : str
        Name of the random variable.  Must match a keyword argument of
        the limit state function.
    dist_obj : scipy.stats.rv_frozen, optional
        A frozen SciPy distribution.  When provided, ``mean`` and
        ``stdv`` are computed from the distribution automatically.
    mean : float, optional
        Mean of the distribution (required if *dist_obj* is ``None``).
    stdv : float, optional
        Standard deviation (required if *dist_obj* is ``None``).
    startpoint : float, optional
        Starting point for iterative search algorithms (defaults to
        the mean).

    Attributes
    ----------
    name : str
        Name of the random variable.
    mean : float
        Mean of the distribution.
    stdv : float
        Standard deviation of the distribution.
    startpoint : float
        Starting point for search algorithms.
    dist_type : str
        Human-readable label set by each subclass (e.g. ``"Normal"``).
    """

    std_normal = StdNormal()

    def __init__(self, name="", dist_obj=None, mean=None, stdv=None, startpoint=None):
        self.name = name
        self.dist_type = "BaseCls"

        # This is the key object that is to be defined in derived classes that
        # are using the base class functionality
        self.dist_obj = dist_obj

        self._update_moments(mean, stdv)
        self.setStartPoint(startpoint)

    def __repr__(self):
        string = self.name + ": " + self.dist_type + " distribution"
        return string

    def _update_moments(self, mean=None, stdv=None):
        if self.dist_obj is not None:
            self.mean = self.dist_obj.mean()
            self.stdv = self.dist_obj.std()
        elif mean is None or stdv is None:
            raise Exception("Mean and std dev must be defined in derived classes")
        else:
            self.mean = mean
            self.stdv = stdv

        if not np.isfinite(self.stdv):
            raise Exception("Std. deviation must be a positive noninfinite number.")

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
        """Probability density function.

        Parameters
        ----------
        x : float or array_like
            Value(s) in physical space.

        Returns
        -------
        float or ndarray
            Density value(s).
        """
        return self.dist_obj.pdf(x)

    def cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : float or array_like
            Value(s) in physical space.

        Returns
        -------
        float or ndarray
            Probability value(s) in [0, 1].
        """
        return self.dist_obj.cdf(x)

    def ppf(self, u):
        """Percent-point function (inverse CDF).

        Parameters
        ----------
        u : float or array_like
            Probability value(s) in (0, 1).

        Returns
        -------
        float or ndarray
            Quantile(s) in physical space.
        """
        return self.dist_obj.ppf(u)

    def u_to_x(self, u):
        """Transform from standard normal space to physical space.

        Applies the marginal Nataf mapping: ``x = F^{-1}(Phi(u))``.

        Parameters
        ----------
        u : float
            Value in standard normal (u) space.

        Returns
        -------
        float
            Corresponding value in physical (x) space.
        """
        return self.dist_obj.ppf(self.std_normal.cdf(u))

    def x_to_u(self, x):
        """Transform from physical space to standard normal space.

        Applies the marginal Nataf mapping: ``u = Phi^{-1}(F(x))``.

        Parameters
        ----------
        x : float
            Value in physical (x) space.

        Returns
        -------
        float
            Corresponding value in standard normal (u) space.
        """
        u = self.std_normal.ppf(self.cdf(x))
        return u

    def jacobian(self, u, x):
        """Diagonal Jacobian of the marginal x-to-u transformation.

        Returns a diagonal matrix ``J`` where the diagonal entry is
        ``f_X(x) / phi(u)`` (Lemaire, eq. 4.9).  This is assembled
        into the full Jacobian by the :class:`Transformation` class.

        Parameters
        ----------
        u : float or array_like
            Value(s) in standard normal space.
        x : float or array_like
            Corresponding value(s) in physical space.

        Returns
        -------
        ndarray
            Diagonal Jacobian matrix of shape ``(n, n)`` where *n* is
            the length of the input arrays.
        """
        pdf1 = self.pdf(x)
        pdf2 = self.std_normal.pdf(u)
        J = np.diag(pdf1 / pdf2)
        return J

    def sample(self, n=1000):
        """Draw random samples from the distribution.

        Uses inverse-transform sampling via ``ppf``.

        Parameters
        ----------
        n : int, optional
            Number of samples (default 1000).

        Returns
        -------
        ndarray
            Array of shape ``(n,)`` with sampled values.
        """
        u = np.random.rand(n)
        samples = self.ppf(u)
        return samples

    def plot(self, ax=None, **kwargs):
        """Plot the probability density function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.  A new figure is created if ``None``.
        **kwargs
            Additional keyword arguments forwarded to
            ``ax.plot()``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        # auto-range
        samples = self.sample()
        x = np.linspace(np.min(samples), np.max(samples), 100)

        show = False
        if ax is None:
            show = True
            _, ax = plt.subplots()

        ax.plot(x, self.pdf(x), label=self.name, **kwargs)
        ax.legend()

        if show:
            plt.show()

        return ax

    # The following must be overidden in derived classes that are not based on a
    # SciPy distribution object, or using the SciPy object for calculations.

    def set_location(self, loc=0):
        """Update the location parameter of the underlying SciPy distribution.

        After updating, ``mean`` and ``stdv`` are recomputed.  This is
        used by the sensitivity analysis to perturb distribution
        parameters.

        Parameters
        ----------
        loc : float, optional
            New location parameter (default 0).

        Raises
        ------
        Exception
            If the distribution does not wrap a SciPy frozen distribution.
        """
        if isinstance(self.dist_obj, rv_frozen):
            pdict = self.dist_obj.kwds
            pdict["loc"] = loc
            self._update_moments()
        else:
            raise Exception("Distribution is not a SciPy object")

    def set_scale(self, scale=1):
        """Update the scale parameter of the underlying SciPy distribution.

        After updating, ``mean`` and ``stdv`` are recomputed.  This is
        used by the sensitivity analysis to perturb distribution
        parameters.

        Parameters
        ----------
        scale : float, optional
            New scale parameter (default 1).

        Raises
        ------
        Exception
            If the distribution does not wrap a SciPy frozen distribution.
        """
        if isinstance(self.dist_obj, rv_frozen):
            pdict = self.dist_obj.kwds
            pdict["scale"] = scale
            self._update_moments()
        else:
            raise Exception("Distribution is not a SciPy object")
