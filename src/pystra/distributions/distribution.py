"""Marginal distribution interfaces and numerically stable tail mappings."""

import warnings
from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Self

from numpy.typing import ArrayLike
from matplotlib.axes import Axes
import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt
from scipy.stats._distn_infrastructure import rv_frozen

from ..errors import ModelError

__all__ = ["StdNormal", "Constant", "Distribution"]


_LOG_HALF = np.log(0.5)
_TINY = np.finfo(float).tiny
_LOG_TINY = np.log(_TINY)
# Above u = 3 the complement 1 - Phi(u) would lose more than 1e-13 of its
# relative precision, so the survival function is used from there on.
_U_SWITCH = 3.0
_P_SWITCH = float(sp.ndtr(_U_SWITCH))


def _log1mexp(a):
    """Return ``log(1 - exp(a))`` for ``a <= 0`` without cancellation.

    Uses ``log(-expm1(a))`` for ``a > -log 2`` and ``log1p(-exp(a))``
    otherwise (Mächler, 2012, "Accurately computing log(1 - exp(-|a|))").
    """
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        return np.where(a > -np.log(2), np.log(-np.expm1(a)), np.log1p(-np.exp(a)))[()]


def _call(function, values):
    """Call a distribution method on an array of points.

    Arrays go to the method at once. A single point is passed as a float,
    and a method written only for scalars (it raises ``TypeError`` for an
    array) is called point by point, so subclass overrides from 1.x keep
    working. Returns a 1-D array.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.array(function(float(values.ravel()[0])), dtype=float, ndmin=1)
    try:
        return np.array(function(values), dtype=float, ndmin=1)
    except TypeError:
        return np.array([function(float(v)) for v in values.ravel()], dtype=float)


def _piecewise(values, upper, lower_fn, upper_fn):
    """Apply ``lower_fn`` where ``upper`` is false and ``upper_fn`` where true.

    Each function receives a 1-D array of the selected values, so only one
    tail formula is evaluated per element. Scalars give NumPy scalars.
    """
    values = np.asarray(values, dtype=float)
    flat = values.ravel()
    upper = np.broadcast_to(upper, values.shape).ravel()
    out = np.empty(flat.shape)
    if not upper.all():
        out[~upper] = _call(lower_fn, flat[~upper])
    if upper.any():
        out[upper] = _call(upper_fn, flat[upper])
    return out.reshape(values.shape)[()]


def _solve_log_tail(log_tail, targets, start, step):
    """Solve ``log_tail(x) = target`` where the tail probability underflows.

    ``log_tail`` is a log-CDF (``step < 0``) or a log-survival function
    (``step > 0``). It exceeds every target at ``start`` and decreases in
    the direction of ``step``. The bracket is widened by doubling, then
    bisected until its ends are adjacent doubles. Points past the support,
    where the logarithm is ``-inf`` or undefined, count as past the
    solution; a solution beyond the double range gives ``+-inf``.
    """
    targets = np.asarray(targets, dtype=float)
    direction = np.sign(step)
    inner = np.full(targets.shape, float(start))
    outer = np.full(targets.shape, direction * np.inf)
    if not np.isfinite(start):
        return outer
    width = abs(float(step))
    todo = np.ones(targets.shape, dtype=bool)
    while todo.any() and np.isfinite(width):
        index = np.flatnonzero(todo)
        trial = inner[index] + direction * width
        with np.errstate(all="ignore"):
            short = log_tail(trial) > targets[index]
        inner[index[short]] = trial[short]
        outer[index[~short]] = trial[~short]
        todo[index[~short]] = False
        width *= 2
    found = np.isfinite(outer)
    lower, upper = inner[found], outer[found]
    for _ in range(1100):
        middle = 0.5 * (lower + upper)
        if not np.any((middle != lower) & (middle != upper)):
            break
        with np.errstate(all="ignore"):
            short = log_tail(middle) > targets[found]
        lower = np.where(short, middle, lower)
        upper = np.where(short, upper, middle)
    outer[found] = 0.5 * (lower + upper)
    return outer


class StdNormal:
    """Standard normal distribution (mean 0, standard deviation 1).

    A lightweight implementation using ``scipy.special`` error functions,
    avoiding the overhead of a full ``scipy.stats`` distribution object.
    This class is used internally by the Nataf transformation and by the
    marginal distribution mappings (``x_to_u`` / ``u_to_x``).
    """

    @staticmethod
    def pdf(u: float | np.ndarray) -> float | np.ndarray:
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
    def cdf(u: ArrayLike) -> float | np.ndarray:
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
        p = sp.ndtr(u)
        return p

    @staticmethod
    def ppf(p: ArrayLike) -> float | np.ndarray:
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
        u = sp.ndtri(p)
        return u


def _uses_native_parameters(distribution, mean, std, **native):
    """Whether a constructor received native parameters instead of moments.

    A distribution takes either its mean and standard deviation or every one
    of its native parameters, never a mixture of the two.
    """
    kind = type(distribution).__name__
    names = " and ".join(native)
    given = [name for name, value in native.items() if value is not None]
    if not given:
        if mean is None or std is None:
            raise TypeError(f"{kind} needs mean and std, or {names}")
        return False
    if mean is not None or std is not None:
        raise TypeError(f"{kind} takes mean and std, or {names}, not both")
    if len(given) < len(native):
        raise TypeError(f"{kind} needs mean and std, or {names}")
    return True


class Constant:
    """A deterministic (non-random) variable in the limit-state function.

    Constants are included in the stochastic model but are not treated as
    random variables — they carry a fixed value through every evaluation of
    the limit-state function.

    Parameters
    ----------
    name : str
        Name of the constant (must match a keyword argument of the limit
        state function).
    value : float
        The fixed value.
    """

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = value

    def get_name(self) -> str:
        """Return the constant name."""
        return self.name

    @property
    def value(self) -> float:
        """The fixed value."""
        return self._value

    def __repr__(self) -> str:
        return f"Constant({self.name!r}, value={self._value!r})"


class Distribution:
    r"""Base class for all probability distributions used in reliability analysis.

    Subclasses typically construct a ``scipy.stats`` frozen distribution
    object (``dist_obj``) and pass it to this base class, which then
    delegates ``pdf``, ``cdf``, ``ppf``, and the Nataf-space
    transformations (``x_to_u``, ``u_to_x``, ``jacobian``) to it.

    Subclasses that do not wrap a SciPy distribution must override the
    transformation and Jacobian methods directly, or provide the tail
    functions below (see, e.g., :class:`Maximum`).

    **Tail accuracy.** Every distribution provides ``cdf``, ``sf``,
    ``logcdf``, ``logsf``, ``pdf``, ``logpdf``, ``ppf`` and ``isf``, each
    accurate in its own tail. ``u_to_x`` and ``x_to_u`` evaluate the tail
    that the point lies in, so a probability near one is never formed:
    ``x = F^{-1}(Phi(u))`` for ``u <= 0`` and ``x = Fbar^{-1}(Phi(-u))``
    for ``u > 0``. Beyond the smallest normal double (``|u|`` above about
    37.5) the log-probability is inverted instead, in closed form where
    one exists and otherwise by solving the log-CDF or log-survival
    function. Jacobians switch to log densities when the densities
    underflow. Subclasses that override ``cdf`` or ``ppf`` should override
    ``sf`` or ``isf`` as well; the fallbacks ``1 - cdf`` and ``ppf(1 - q)``
    lose the upper tail.

    Parameters
    ----------
    name : str
        Name of the random variable.  Must match a keyword argument of
        the limit-state function.
    dist_obj : scipy.stats.rv_frozen, optional
        A frozen SciPy distribution.  When provided, ``mean`` and
        ``std`` are computed from the distribution automatically.
    mean : float, optional
        Mean of the distribution (required if *dist_obj* is ``None``).
    std : float, optional
        Standard deviation (required if *dist_obj* is ``None``).
    start_point : float, optional
        Starting point for iterative search algorithms (defaults to
        the mean).

    Attributes
    ----------
    name : str
        Name of the random variable.
    dist_type : str
        Human-readable label set by each subclass (e.g. ``"Normal"``).
    """

    std_normal = StdNormal()

    def __init__(
        self,
        name: str = "",
        dist_obj: rv_frozen | None = None,
        mean: float | None = None,
        std: float | None = None,
        start_point: float | None = None,
    ) -> None:
        self.name = name
        self.dist_type = "BaseCls"

        # This is the key object that is to be defined in derived classes that
        # are using the base class functionality
        self.dist_obj = dist_obj

        self._update_moments(mean, std)
        self._set_start_point(start_point)

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return self._mean

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return self._std

    @property
    def start_point(self) -> float:
        """Starting point of the design-point search; the mean by default."""
        return self._start_point

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, mean={self.mean:.6g}, std={self.std:.6g})"

    def _update_moments(self, mean=None, std=None):
        if self.dist_obj is not None:
            self._mean = self.dist_obj.mean()
            self._std = self.dist_obj.std()
        elif mean is None or std is None:
            raise ModelError("Mean and std dev must be defined in derived classes")
        else:
            self._mean = mean
            self._std = std

        if not np.isfinite(self.std) or self.std <= 0:
            raise ModelError("Std. deviation must be a positive noninfinite number.")

    def get_name(self) -> str:
        """Return the random-variable name."""
        return self.name

    def _set_start_point(self, start_point=None):
        if start_point is None:
            self._start_point = self.mean
        else:
            self._start_point = start_point

    # The following can be overridden by derived classes to implement more
    # efficient calculations where desirable

    def pdf(self, x: ArrayLike) -> float | np.ndarray:
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

    def logpdf(self, x: ArrayLike) -> float | np.ndarray:
        """Log density, ``-inf`` where the density is zero."""
        if self.dist_obj is not None and type(self).pdf is Distribution.pdf:
            return self.dist_obj.logpdf(x)
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x))

    def cdf(self, x: ArrayLike) -> float | np.ndarray:
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

    def sf(self, x: ArrayLike) -> float | np.ndarray:
        """Survival function ``1 - F(x)``, accurate in the upper tail."""
        if self._scipy_cdf():
            return self.dist_obj.sf(x)
        return 1 - self.cdf(x)

    def logcdf(self, x: ArrayLike) -> float | np.ndarray:
        """Logarithm of the CDF, accurate in both tails.

        Below the median the logarithm of the CDF itself (or SciPy's
        ``logcdf``) is used; above it, ``log1p(-sf(x))`` keeps the
        difference from zero.
        """
        c = np.asarray(self.cdf(x), dtype=float)
        return _piecewise(
            x, c > 0.5, self._lower_logcdf, lambda v: np.log1p(-self.sf(v))
        )

    def logsf(self, x: ArrayLike) -> float | np.ndarray:
        """Logarithm of the survival function, accurate in both tails."""
        c = np.asarray(self.cdf(x), dtype=float)
        return _piecewise(
            x, c > 0.5, lambda v: np.log1p(-self.cdf(v)), self._upper_logsf
        )

    def _lower_logcdf(self, x):
        """Log-CDF below the median."""
        if self._scipy_cdf():
            return self.dist_obj.logcdf(x)
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x))

    def _upper_logsf(self, x):
        """Log-survival function above the median."""
        if self._scipy_cdf():
            return self.dist_obj.logsf(x)
        with np.errstate(divide="ignore"):
            return np.log(self.sf(x))

    def _scipy_cdf(self):
        """Whether the CDF family can be delegated to ``dist_obj``."""
        return self.dist_obj is not None and type(self).cdf is Distribution.cdf

    def ppf(self, u: ArrayLike) -> float | np.ndarray:
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

    def isf(self, q: ArrayLike) -> float | np.ndarray:
        """Inverse survival function, accurate for small upper-tail ``q``."""
        if self.dist_obj is not None and type(self).ppf is Distribution.ppf:
            return self.dist_obj.isf(q)
        return self.ppf(1 - np.asarray(q, dtype=float))

    def _ppf_log(self, logp):
        """Quantile at which ``logcdf`` equals ``logp``, for any ``logp <= 0``."""
        logp = np.asarray(logp, dtype=float)
        return _piecewise(
            logp,
            logp > _LOG_HALF,
            self._lower_quantile_log,
            lambda v: self._upper_quantile_log(_log1mexp(v)),
        )

    def _isf_log(self, logq):
        """Quantile at which ``logsf`` equals ``logq``, for any ``logq <= 0``."""
        logq = np.asarray(logq, dtype=float)
        return _piecewise(
            logq,
            logq > _LOG_HALF,
            self._upper_quantile_log,
            lambda v: self._lower_quantile_log(_log1mexp(v)),
        )

    def _lower_quantile_log(self, logp):
        """Lower-tail quantile from a log-probability of at most log(1/2).

        Below the smallest normal double the log-CDF is solved directly.
        Subclasses with a closed form override this.
        """
        logp = np.atleast_1d(np.asarray(logp, dtype=float))
        with np.errstate(under="ignore"):
            x = _call(self.ppf, np.exp(logp))
        deep = (logp < _LOG_TINY) & np.isfinite(logp)
        if deep.any():
            x[deep] = _solve_log_tail(
                self.logcdf, logp[deep], self.ppf(_TINY), -self.std
            )
        return x

    def _upper_quantile_log(self, logq):
        """Upper-tail quantile from a log-probability of at most log(1/2)."""
        logq = np.atleast_1d(np.asarray(logq, dtype=float))
        with np.errstate(under="ignore"):
            x = _call(self.isf, np.exp(logq))
        deep = (logq < _LOG_TINY) & np.isfinite(logq)
        if deep.any():
            x[deep] = _solve_log_tail(self.logsf, logq[deep], self.isf(_TINY), self.std)
        return x

    def u_to_x(self, u: ArrayLike) -> float | np.ndarray:
        """Transform from standard normal space to physical space.

        Applies the marginal Nataf mapping ``x = F^{-1}(Phi(u))``. Above
        ``u = 3`` it is evaluated as ``Fbar^{-1}(Phi(-u))``, so the
        probability passed on is never rounded towards one; far into either
        tail the log-probability is used (see :meth:`_tail_quantiles`).

        Parameters
        ----------
        u : float or array_like
            Value(s) in standard normal (u) space.

        Returns
        -------
        float or ndarray
            Corresponding value(s) in physical (x) space.
        """
        u = np.asarray(u, dtype=float)
        if u.ndim == 0:
            value = float(u)
            upper = value > _U_SWITCH
            probability = float(sp.ndtr(-value if upper else value))
            if probability >= 1e-8:
                quantile = self.isf if upper else self.ppf
                return np.asarray(quantile(probability), dtype=float).reshape(())[()]
        shape = u.shape
        u = u.ravel()
        p = sp.ndtr(u)
        upper = np.flatnonzero(u > _U_SWITCH)
        lower = np.flatnonzero(u < -5.0)
        central = ~((u > _U_SWITCH) | (u < -5.0))
        x = np.empty(u.shape)
        if central.any():
            x[central] = _call(self.ppf, p[central])
        for index, sign, quantile, log_tail, log_quantile in (
            (upper, -1, self.isf, self.logsf, self._upper_quantile_log),
            (lower, 1, self.ppf, self.logcdf, self._lower_quantile_log),
        ):
            if index.size:
                z = sign * u[index]
                prob = sp.ndtr(z)
                with warnings.catch_warnings():
                    # SciPy warns when its numerical inverse gives up; that
                    # result is checked and solved again
                    warnings.simplefilter("ignore", RuntimeWarning)
                    xt = _call(quantile, prob)
                    x[index] = self._tail_quantiles(
                        xt, prob, z, log_quantile, log_tail, -sign
                    )
        return x.reshape(shape)[()]

    def _tail_quantiles(self, x, prob, z, quantile_log, log_tail, direction):
        """Refine quantiles whose tail probability is below 1e-8.

        ``prob = Phi(z)`` is the tail probability of each point. Below the
        smallest normal double the quantile comes from ``log Phi(z)``.
        Numerical inverse CDFs (SciPy's beta, for example) can also stall
        or return NaN far into a tail, so each tail quantile is checked
        against the log-CDF or log-survival function and solved again if
        it misses by more than 1e-8 in relative log-probability.
        """
        index = np.flatnonzero(prob < 1e-8)
        if index.size == 0:
            return x
        target = sp.log_ndtr(z[index])
        deep = prob[index] < _TINY
        if deep.any():
            x[index[deep]] = quantile_log(target[deep])
        with np.errstate(all="ignore"):
            got = _call(log_tail, x[index])
        bad = ~(np.abs(got - target) <= 1e-8 * np.abs(target))
        if bad.any():
            x[index[bad]] = _solve_log_tail(
                log_tail, target[bad], self.ppf(0.5), direction * self.std
            )
        return x

    def x_to_u(self, x: ArrayLike) -> float | np.ndarray:
        """Transform from physical space to standard normal space.

        Applies the marginal Nataf mapping ``u = Phi^{-1}(F(x))``, as
        ``u = -Phi^{-1}(Fbar(x))`` above ``u = 3``, with log probabilities
        where these underflow.

        Parameters
        ----------
        x : float or array_like
            Value(s) in physical (x) space.

        Returns
        -------
        float or ndarray
            Corresponding value(s) in standard normal (u) space.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            value = float(x)
            probability = float(np.asarray(self.cdf(value)).item())
            if probability > _P_SWITCH:
                survival = float(np.asarray(self.sf(value)).item())
                if survival < _TINY:
                    return -sp.ndtri_exp(float(np.asarray(self.logsf(value)).item()))
                return -sp.ndtri(survival)
            if probability < _TINY:
                return sp.ndtri_exp(float(np.asarray(self.logcdf(value)).item()))
            return sp.ndtri(probability)
        shape = x.shape
        x = x.ravel()
        c = _call(self.cdf, x)
        u = sp.ndtri(c)
        deep = c < _TINY
        if deep.any():
            u[deep] = sp.ndtri_exp(_call(self.logcdf, x[deep]))
        upper = c > _P_SWITCH
        if upper.any():
            s = _call(self.sf, x[upper])
            v = -sp.ndtri(s)
            tail = s < _TINY
            if tail.any():
                v[tail] = -sp.ndtri_exp(_call(self.logsf, x[upper][tail]))
            u[upper] = v
        return u.reshape(shape)[()]

    def jacobian(self, u: ArrayLike, x: ArrayLike) -> np.ndarray:
        """Diagonal Jacobian of the marginal x-to-u transformation.

        Returns a diagonal matrix ``J`` where the diagonal entry is
        ``f_X(x) / phi(u)`` (Lemaire, eq. 4.9).  This is assembled
        into the full Jacobian by the :class:`Transformation` class.
        The ratio is formed from log densities where either density
        underflows.

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
        u = np.atleast_1d(np.asarray(u, dtype=float))
        x = np.atleast_1d(np.asarray(x, dtype=float))
        pdf1 = np.array(self.pdf(x), dtype=float, ndmin=1)
        pdf2 = self.std_normal.pdf(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = pdf1 / pdf2
        tail = (pdf1 < 1e-300) | (pdf2 < 1e-300)
        if tail.any():
            log_phi = -0.5 * u[tail] ** 2 - 0.5 * np.log(2 * np.pi)
            ratio[tail] = np.exp(self.logpdf(x[tail]) - log_phi)
        J = np.diag(ratio)
        return J

    def sample(self, n: int = 1000) -> np.ndarray:
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

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
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
    def sensitivity_params(self) -> dict[str, float]:
        r"""Distribution parameters for which sensitivities are computed.

        Returns a dict ``{param_name: current_value}`` listing every
        parameter with respect to which :math:`\partial\beta/\partial\theta`
        should be evaluated.

        The default implementation returns ``{"mean": μ, "std": σ}``,
        which is appropriate for most distributions.  Distributions with
        additional parameters of interest (e.g. the GEV shape parameter)
        should override this property to include them.

        Fixed constructor settings such as bounds are separate from these
        sensitivity coordinates. Native reconstruction parameters can be
        another representation of the same law: sensitivity analysis perturbs
        the declared coordinates together, keeping other moments fixed.

        Returns
        -------
        dict
            ``{param_name: current_value}``
        """
        return {"mean": self.mean, "std": self.std}

    # Native coordinates replaced when the caller supplies physical moments.
    _native_parameters = ()

    def _parameter_values(self):
        return {"mean": self.mean, "std": self.std}

    @property
    def parameters(self) -> Mapping[str, object]:
        """Read-only snapshot of keyword arguments that rebuild this marginal.

        ``type(dist)(**dist.parameters)`` reproduces its law, name and start
        point. Built-ins use native parameters where moment fitting could
        lose precision. Nested marginals and SciPy objects are independent
        copies, so mutating a snapshot never changes the source distribution.
        Custom marginals may override this property with their complete
        constructor mapping. Sensitivity coordinates are specified separately
        by :attr:`sensitivity_params`.
        """
        return MappingProxyType(
            deepcopy(
                {
                    "name": self.name,
                    **self._parameter_values(),
                    "start_point": self.start_point,
                }
            )
        )

    def with_parameters(self, **changes: object) -> Self:
        """Rebuild an independent marginal, replacing constructor parameters.

        Parameters
        ----------
        **changes
            Replacements for keys of :attr:`parameters`. Built-ins supporting
            moment sensitivities also accept ``mean`` and ``std``; supplying
            either switches to moment construction, holding the other moment
            and fixed bounds/shape constant. Native coordinates and moments
            cannot be supplied together. The start point remains unchanged
            unless explicitly replaced (``None`` selects the new mean).

        Returns
        -------
        Distribution
            Independent instance of the same type, including nested marginals.
            An empty replacement reproduces the distribution.

        Raises
        ------
        TypeError
            A replacement is unknown, or mixes moments and native coordinates.
        ModelError, ValueError
            Constructor parameters do not define a valid distribution.
        """
        parameters = dict(self.parameters)
        allowed = parameters.keys() | self.sensitivity_params.keys()
        unknown = changes.keys() - allowed
        if unknown:
            raise TypeError(
                f"Unknown distribution parameters: {', '.join(sorted(unknown))}"
            )
        if changes.keys() & {"mean", "std"} and self._native_parameters:
            if changes.keys() & set(self._native_parameters):
                raise TypeError("Specify moments or native parameters, not both")
            for key in self._native_parameters:
                parameters.pop(key, None)
            parameters.update(mean=self.mean, std=self.std)
        parameters.update(changes)
        return type(self)(**deepcopy(parameters))

    def _dmoments_dtheta(self, param):
        r"""Derivatives of mean and standard deviation w.r.t. a parameter.

        Returns ``(∂μ/∂θ, ∂σ/∂θ)`` for the parameter named *param*.
        This is needed by :func:`~pystra._numerics.integration.drho0_dtheta` to
        evaluate the general form of :math:`\partial h/\partial\theta`
        (Eq. 24 of Bourinet 2017).

        For ``"mean"`` and ``"std"`` the derivatives are exact:
        ``(1, 0)`` and ``(0, 1)`` respectively.  For any other parameter
        (e.g. a shape parameter) central finite differences via
        :meth:`with_parameters` are used.

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
        d_plus = self.with_parameters(**{**self.sensitivity_params, param: val + h})
        d_minus = self.with_parameters(**{**self.sensitivity_params, param: val - h})
        return (
            (d_plus.mean - d_minus.mean) / (2 * h),
            (d_plus.std - d_minus.std) / (2 * h),
        )

    def cdf_gradient(self, x: ArrayLike) -> dict[str, float | np.ndarray]:
        r"""Derivatives of the CDF w.r.t. each sensitivity parameter.

        Returns ``∂F_X(x)/∂θ`` for every parameter listed by
        :attr:`sensitivity_params`.  The base-class implementation uses
        central finite differences on the CDF via :meth:`with_parameters`.

        Before computing derivatives, a reconstruction sanity check
        verifies that :meth:`with_parameters` (with no overrides) reproduces
        the current distribution.  This catches both constructor
        failures (e.g. composite distributions) and silent mismatches
        (e.g. distributions whose extra constructor arguments are not
        stored in :attr:`parameters`).

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
            :meth:`with_parameters`.
        """
        if not self.sensitivity_params:
            raise ValueError(
                f"{type(self).__name__} does not support sensitivity analysis"
            )
        # Validate that reconstruction reproduces this distribution.
        try:
            test = self.with_parameters()
        except Exception as e:
            raise ValueError(
                f"{type(self).__name__} does not support sensitivity "
                f"analysis.  Define parameters on the subclass "
                f"or override with_parameters()."
            ) from e
        # Use a scalar test point for validation (x may be an array
        # when called from drho0_dtheta with quadrature grids)
        x_test = float(self.mean + 0.5 * self.std)
        ref_cdf = float(self.cdf(x_test))
        test_cdf = float(test.cdf(x_test))
        if abs(test_cdf - ref_cdf) > 1e-6 * (1 + abs(ref_cdf)):
            raise ValueError(
                f"{type(self).__name__}.with_parameters() does not faithfully "
                f"reconstruct the distribution (CDF mismatch at "
                f"x={x_test}: original={ref_cdf:.8g}, "
                f"copy={test_cdf:.8g}).  "
                f"Define a complete parameters mapping on the subclass."
            )

        result = {}
        for param, val in self.sensitivity_params.items():
            h = max(abs(val) * 1e-6, self.std * 1e-8)
            d_plus = self.with_parameters(**{**self.sensitivity_params, param: val + h})
            d_minus = self.with_parameters(
                **{**self.sensitivity_params, param: val - h}
            )
            result[param] = (d_plus.cdf(x) - d_minus.cdf(x)) / (2 * h)
        return result

    def set_location(self, loc: float = 0) -> None:
        """Update the location parameter of the underlying SciPy distribution.

        After updating, ``mean`` and ``std`` are recomputed.  This is
        available for callers that need an in-place parameter update.

        Parameters
        ----------
        loc : float, optional
            New location parameter (default 0).

        Raises
        ------
        ModelError
            If the distribution does not wrap a SciPy frozen distribution.
        """
        if isinstance(self.dist_obj, rv_frozen):
            pdict = self.dist_obj.kwds
            pdict["loc"] = loc
            self._update_moments()
        else:
            raise ModelError("Distribution is not a SciPy object")

    def set_scale(self, scale: float = 1) -> None:
        """Update the scale parameter of the underlying SciPy distribution.

        After updating, ``mean`` and ``std`` are recomputed.  This is
        available for callers that need an in-place parameter update.

        Parameters
        ----------
        scale : float, optional
            New scale parameter (default 1).

        Raises
        ------
        ModelError
            If the distribution does not wrap a SciPy frozen distribution.
        """
        if isinstance(self.dist_obj, rv_frozen):
            pdict = self.dist_obj.kwds
            pdict["scale"] = scale
            self._update_moments()
        else:
            raise ModelError("Distribution is not a SciPy object")
