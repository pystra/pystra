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

        # Extra constructor keyword arguments needed to faithfully
        # reconstruct this distribution beyond (name, mean, stdv).
        # Subclasses with additional parameters (e.g. shape for GEV,
        # bounds for Beta) should set this in their __init__ *before*
        # calling super().__init__().  These are passed through by
        # _make_copy() but are NOT sensitivity parameters — they are
        # treated as fixed constants unless the subclass also adds them
        # to sensitivity_params.
        if not hasattr(self, "_ctor_kwargs"):
            self._ctor_kwargs = {}

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

    # ------------------------------------------------------------------
    # Sensitivity-analysis support
    # ------------------------------------------------------------------

    @property
    def sensitivity_params(self):
        r"""Distribution parameters for which sensitivities are computed.

        Returns a dict ``{param_name: current_value}`` listing every
        parameter with respect to which :math:`\partial\beta/\partial\theta`
        should be evaluated.

        The default implementation returns ``{"mean": μ, "std": σ}``,
        which is appropriate for most distributions.  Distributions with
        additional parameters of interest (e.g. the GEV shape parameter)
        should override this property to include them.

        Parameters listed in :attr:`_ctor_kwargs` but **not** in
        ``sensitivity_params`` are held fixed during sensitivity analysis
        — they are only used by :meth:`_make_copy` to faithfully
        reconstruct the distribution.

        Returns
        -------
        dict
            ``{param_name: current_value}``
        """
        return {"mean": self.mean, "std": self.stdv}

    def _make_copy(self, **overrides):
        r"""Construct a copy of this distribution with perturbed parameters.

        Builds a fresh instance of the same type using the current
        ``(mean, stdv)`` and any extra constructor keyword arguments
        stored in :attr:`_ctor_kwargs`.  The *overrides* dict replaces
        individual parameter values; its keys must match those of
        :attr:`sensitivity_params` or :attr:`_ctor_kwargs`.

        Parameters
        ----------
        **overrides
            Parameter values to override.  For example,
            ``dist._make_copy(mean=dist.mean + h)`` perturbs the mean.

        Returns
        -------
        Distribution
            A new distribution instance with the perturbed parameters.

        Raises
        ------
        TypeError
            If the subclass constructor does not accept the provided
            arguments (e.g. a composite distribution that cannot be
            reconstructed from ``(name, mean, stdv)``).
        """
        params = {"mean": self.mean, "std": self.stdv}
        params.update(self._ctor_kwargs)
        params.update(overrides)
        mean = params.pop("mean")
        stdv = params.pop("std")
        return type(self)(self.name, mean, stdv, **params)

    def _dmoments_dtheta(self, param):
        r"""Derivatives of mean and standard deviation w.r.t. a parameter.

        Returns ``(∂μ/∂θ, ∂σ/∂θ)`` for the parameter named *param*.
        This is needed by :func:`~pystra.integration.drho0_dtheta` to
        evaluate the general form of :math:`\partial h/\partial\theta`
        (Eq. 24 of Bourinet 2017).

        For ``"mean"`` and ``"std"`` the derivatives are exact:
        ``(1, 0)`` and ``(0, 1)`` respectively.  For any other parameter
        (e.g. a shape parameter) central finite differences via
        :meth:`_make_copy` are used.

        Parameters
        ----------
        param : str
            Parameter name (a key of :attr:`sensitivity_params`).

        Returns
        -------
        tuple of float
            ``(∂μ/∂θ, ∂σ/∂θ)``
        """
        if param == "mean":
            return (1.0, 0.0)
        elif param == "std":
            return (0.0, 1.0)
        val = self.sensitivity_params[param]
        h = max(abs(val) * 1e-6, 1e-10)
        d_plus = self._make_copy(**{param: val + h})
        d_minus = self._make_copy(**{param: val - h})
        return (
            (d_plus.mean - d_minus.mean) / (2 * h),
            (d_plus.stdv - d_minus.stdv) / (2 * h),
        )

    def dF_dtheta(self, x):
        r"""Derivatives of the CDF w.r.t. each sensitivity parameter.

        Returns ``∂F_X(x)/∂θ`` for every parameter listed by
        :attr:`sensitivity_params`.  The base-class implementation uses
        central finite differences on the CDF via :meth:`_make_copy`.

        Before computing derivatives, a reconstruction sanity check
        verifies that :meth:`_make_copy` (with no overrides) reproduces
        the current distribution.  This catches both constructor
        failures (e.g. composite distributions) and silent mismatches
        (e.g. distributions whose extra constructor arguments are not
        stored in :attr:`_ctor_kwargs`).

        Subclasses may override this with analytical expressions for
        better accuracy and performance (see :class:`Normal` and
        :class:`Lognormal`).

        Parameters
        ----------
        x : float
            Evaluation point in physical space.

        Returns
        -------
        dict
            ``{param_name: ∂F/∂θ}`` for each parameter in
            :attr:`sensitivity_params`.

        Raises
        ------
        ValueError
            If the distribution cannot be faithfully reconstructed by
            :meth:`_make_copy`.
        """
        # --- Validate that _make_copy reproduces this distribution ---
        try:
            test = self._make_copy()
        except Exception as e:
            raise ValueError(
                f"{type(self).__name__} does not support sensitivity "
                f"analysis.  Set _ctor_kwargs in the subclass __init__ "
                f"or override _make_copy()."
            ) from e
        # Use a scalar test point for validation (x may be an array
        # when called from drho0_dtheta with quadrature grids)
        x_test = float(self.mean + 0.5 * self.stdv)
        ref_cdf = float(self.cdf(x_test))
        test_cdf = float(test.cdf(x_test))
        if abs(test_cdf - ref_cdf) > 1e-6 * (1 + abs(ref_cdf)):
            raise ValueError(
                f"{type(self).__name__}._make_copy() does not faithfully "
                f"reconstruct the distribution (CDF mismatch at "
                f"x={x_test}: original={ref_cdf:.8g}, "
                f"copy={test_cdf:.8g}).  "
                f"Set _ctor_kwargs correctly in the subclass __init__."
            )

        result = {}
        for param, val in self.sensitivity_params.items():
            h = max(abs(val) * 1e-6, self.stdv * 1e-8)
            d_plus = self._make_copy(**{param: val + h})
            d_minus = self._make_copy(**{param: val - h})
            result[param] = (d_plus.cdf(x) - d_minus.cdf(x)) / (2 * h)
        return result

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
