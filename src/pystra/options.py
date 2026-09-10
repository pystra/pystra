"""Immutable settings for reliability analyses.

Each analysis takes one frozen settings object through its ``options``
keyword: :class:`FORMOptions` for FORM and the design-point methods built on
it (system FORM, sensitivity analysis, the Strong Maximum Test),
:class:`SORMOptions` for SORM, and :class:`SimulationOptions` for the
simulation methods and active learning. Unknown fields are rejected, values
are validated when the object is created, and the defaults are those of
PySTRA 1.x. Use :func:`dataclasses.replace` to derive modified settings.

A simulation method rejects a :class:`SimulationOptions` setting that it
does not use, rather than silently ignoring it.
"""

import math
from dataclasses import dataclass, field, fields
from numbers import Integral, Real
from typing import Optional, Tuple

from .errors import ModelError

__all__ = ["FORMOptions", "SORMOptions", "SimulationOptions"]

TRANSFORMS = (None, "cholesky", "svd", "nataf", "rosenblatt")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ModelError(f"{name} must be a positive integer")


def _positive(value, name, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not zero)
    ):
        bound = "non-negative" if zero else "positive"
        raise ModelError(f"{name} must be a finite {bound} number")


def _choice(value, name, choices):
    if value not in choices:
        raise ModelError(f"{name} must be one of {choices}, not {value!r}")


def _dependence(options):
    _choice(options.transform, "transform", TRANSFORMS)
    order = options.rosenblatt_order
    if order is not None:
        order = tuple(order)
        if not all(isinstance(i, Integral) and not isinstance(i, bool) for i in order):
            raise ModelError("rosenblatt_order must be a sequence of variable indices")
        object.__setattr__(options, "rosenblatt_order", tuple(int(i) for i in order))


@dataclass(frozen=True, kw_only=True)
class FORMOptions:
    """Settings for FORM and the design-point methods built on it.

    Parameters
    ----------
    max_iterations : int, default 100
        Iteration limit of the design-point search.
    limit_state_tolerance : float, default 1e-3
        Convergence tolerance on :math:`|g(u)| / |g(u_0)|`, how close the
        point is to the limit-state surface.
    gradient_tolerance : float, default 1e-3
        Convergence tolerance on how nearly the gradient points towards the
        origin, :math:`\\|u - (\\alpha \\cdot u)\\alpha\\|`.
    step_size : float, default 0
        Fixed step length, or 0 to choose each step by the Armijo rule.
    differentiation : {"ffd", "ddm"}, default "ffd"
        Forward finite differences, or direct differentiation with a
        gradient returned by the limit-state function.
    ffd_parameter : float, default 1000
        Finite-difference step divisor: each variable is perturbed by its
        standard deviation divided by this value. 1000 suits closed-form
        limit states; about 50 suits finite-element models.
    block_size : int, default 1000
        Points passed to the limit-state function per call.
    transform : {None, "cholesky", "svd", "nataf", "rosenblatt"}, default None
        Isoprobabilistic transformation; None selects it from the model.
    rosenblatt_order : sequence of int, optional
        Conditioning order of the variables for the Rosenblatt
        transformation.
    """

    max_iterations: int = 100
    limit_state_tolerance: float = 1e-3
    gradient_tolerance: float = 1e-3
    step_size: float = 0
    differentiation: str = "ffd"
    ffd_parameter: float = 1000
    block_size: int = 1000
    transform: Optional[str] = None
    rosenblatt_order: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        _integer(self.max_iterations, "max_iterations")
        _positive(self.limit_state_tolerance, "limit_state_tolerance")
        _positive(self.gradient_tolerance, "gradient_tolerance")
        _positive(self.step_size, "step_size", zero=True)
        _choice(self.differentiation, "differentiation", ("ffd", "ddm"))
        _positive(self.ffd_parameter, "ffd_parameter")
        _integer(self.block_size, "block_size")
        _dependence(self)


@dataclass(frozen=True, kw_only=True)
class SORMOptions:
    """Settings for SORM.

    Parameters
    ----------
    fit : {"curve", "point"}, default "curve"
        Curve fitting through the Hessian at the design point, or point
        fitting on the limit-state surface.
    formula : {"breitung", "modified_breitung"}, default "breitung"
        The approximation reported as the estimate; both are computed.
    ffd_parameter : float, default 1000
        Finite-difference step divisor for the Hessian: the step in standard
        coordinates is its reciprocal.
    form : FORMOptions, default FORMOptions()
        Settings of the FORM analysis that SORM runs when none is supplied.
        A supplied FORM analysis keeps its own settings.
    """

    fit: str = "curve"
    formula: str = "breitung"
    ffd_parameter: float = 1000
    form: FORMOptions = field(default_factory=FORMOptions)

    def __post_init__(self):
        _choice(self.fit, "fit", ("curve", "point"))
        _choice(self.formula, "formula", ("breitung", "modified_breitung"))
        _positive(self.ffd_parameter, "ffd_parameter")
        if not isinstance(self.form, FORMOptions):
            raise ModelError("form must be FORMOptions")


@dataclass(frozen=True, kw_only=True)
class SimulationOptions:
    """Settings for the simulation methods and active learning.

    Parameters
    ----------
    n_samples : int, default 100000
        Sample budget: points for crude Monte Carlo, importance sampling and
        distribution analysis, lines for line sampling, and samples per level
        for subset simulation.
    block_size : int, default 1000
        Points passed to the limit-state function per call, and crude Monte
        Carlo's convergence-check interval.
    target_cov : float, default 0.05
        Crude Monte Carlo and importance sampling stop once the estimate's
        coefficient of variation reaches this value; 0 uses the whole budget.
    sampling_std : float, default 1.0
        Standard deviation of the sampling density in standard coordinates.
    bins : int, optional
        Histogram bins for distribution analysis; chosen from the sample size
        if omitted.
    transform : {None, "cholesky", "svd", "nataf", "rosenblatt"}, default None
        Isoprobabilistic transformation; None selects it from the model.
    rosenblatt_order : sequence of int, optional
        Conditioning order of the variables for the Rosenblatt
        transformation.
    """

    n_samples: int = 100_000
    block_size: int = 1000
    target_cov: float = 0.05
    sampling_std: float = 1.0
    bins: Optional[int] = None
    transform: Optional[str] = None
    rosenblatt_order: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        _integer(self.n_samples, "n_samples")
        _integer(self.block_size, "block_size")
        _positive(self.target_cov, "target_cov", zero=True)
        _positive(self.sampling_std, "sampling_std")
        if self.bins is not None:
            _integer(self.bins, "bins")
        _dependence(self)

    def _require_defaults(self, method, unused):
        """Reject settings that *method* does not use."""
        defaults = SimulationOptions()
        changed = [
            name for name in unused if getattr(self, name) != getattr(defaults, name)
        ]
        if changed:
            raise ModelError(f"{method} does not use {', '.join(changed)}")
