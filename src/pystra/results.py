r"""Immutable records returned by every reliability analysis.

Each analysis's ``run()`` returns one of these frozen records. A record is a
snapshot: rerunning the analysis, or changing its model, does not alter it.
Vector fields are read-only NumPy arrays in ``variable_names`` order, and
mapping fields are read-only. Records compare equal when their values are
equal.

Every record has ``method``, ``status``, ``message``,
``n_limit_state_evaluations``, ``variable_names`` and ``options``, the
``converged`` property and :meth:`~FORMResult.summary`. Records of probability
estimates add ``failure_probability`` and ``beta``, the normal-equivalent
reliability index :math:`-\Phi^{-1}(p_f)`.

``status`` is one of:

``"converged"``
    A design-point method (FORM, SORM, system FORM, sensitivity) met its
    convergence criteria.
``"not_converged"``
    It did not, or SORM's formula is undefined at the fitted curvatures.
    The estimate is unavailable or unreliable. Unless constructed with
    ``on_failure="return"``, the analysis raises :class:`~pystra.AnalysisError`
    carrying the record.
``"completed"``
    A simulation or diagnostic ran to completion.
``"precision_not_met"``
    A simulation used its whole sample budget before reaching its target
    coefficient of variation. The estimate is still reported.

``n_limit_state_evaluations`` counts the limit-state evaluations made by the
method itself. A nested FORM record (``form``, ``component_results`` or the
``form`` diagnostic) reports that FORM analysis's own count; system FORM and
sensitivity analysis, which consist of FORM runs, report their total.
``options`` holds the frozen settings the analysis used.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Optional

import numpy as np
from pandas import DataFrame

import pystra as _pystra

__all__ = [
    "FORMResult",
    "SORMResult",
    "SimulationResult",
    "SystemFORMResult",
    "SensitivityResult",
    "StrongMaximumResult",
    "DistributionAnalysisResult",
]

STATUSES = ("converged", "not_converged", "completed", "precision_not_met")


class _ReadOnlyMapping(Mapping):
    """A read-only mapping that, unlike ``MappingProxyType``, can be pickled."""

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(self._data)


def _frozen(value):
    """Copy *value* with read-only arrays and mappings and tuple sequences."""
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.flags.writeable = False
        return value
    if isinstance(value, Mapping):
        return _ReadOnlyMapping({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _same(a, b):
    """Compare field values, element by element for arrays and mappings."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        )
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(map(_same, a, b))
    return a is b or a == b


def _text(value):
    if value is None:
        return "unavailable"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


@dataclass(frozen=True, kw_only=True, eq=False)
class _Result:
    """Fields and behavior common to every analysis record."""

    method: str
    status: str
    message: str
    n_limit_state_evaluations: int
    variable_names: tuple
    options: Any = field(default=None, repr=False)

    _arrays: ClassVar[tuple] = ()
    _read_only: ClassVar[tuple] = ()

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"Unknown status {self.status!r}; use one of {STATUSES}")
        object.__setattr__(self, "variable_names", tuple(self.variable_names))
        for name in self._arrays:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _frozen(np.array(value, dtype=float)))
        for name in self._read_only:
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return all(
            _same(getattr(self, f.name), getattr(other, f.name)) for f in fields(self)
        )

    __hash__ = None

    @property
    def converged(self) -> bool:
        """Whether the method met its termination criteria."""
        return self.status in ("converged", "completed")

    def _rows(self):
        return []

    def summary(self) -> str:
        """Return a plain-text report of the principal results."""
        rows = [
            ("Status", self.status),
            ("Message", self.message),
            *self._rows(),
            ("Limit-state evaluations", self.n_limit_state_evaluations),
        ]
        width = max(len(label) for label, _ in rows)
        lines = [f"  {label:<{width}}  {_text(value)}" for label, value in rows]
        return "\n".join([f"{self.method} result", *lines])


@dataclass(frozen=True, kw_only=True, eq=False)
class _ProbabilityResult(_Result):
    """A record of a failure-probability estimate."""

    failure_probability: Optional[float]
    beta: Optional[float]

    def _rows(self):
        return [
            ("Failure probability", self.failure_probability),
            ("Reliability index", self.beta),
        ]


@dataclass(frozen=True, kw_only=True, eq=False)
class FORMResult(_ProbabilityResult):
    """Record of a FORM analysis.

    ``beta`` is normal-equivalent, :math:`-\\Phi^{-1}(p_f)`. ``design_index``
    is the signed distance of the design point from the origin in
    ``standard_space``; the two are equal in normal space and differ in
    Student-t standard space.

    Attributes
    ----------
    design_index : float or None
        Signed distance of the design point in ``standard_space``.
    design_point_x : ndarray or None
        Design point in physical coordinates and original units.
    design_point_u : ndarray or None
        The same point in standard coordinates.
    alpha : ndarray or None
        Unit vector from the origin towards the design point in standard
        coordinates, the FORM importance direction.
    standard_space : str
        ``"normal"`` or ``"student_t"``.
    iterations : int
        Iterations performed.
    limit_state_error, direction_error : float or None
        The final convergence residuals.

    An unconverged record has no probability, index, design point or
    direction; its diagnostics remain available.
    """

    design_index: Optional[float]
    design_point_x: Optional[np.ndarray]
    design_point_u: Optional[np.ndarray]
    alpha: Optional[np.ndarray]
    standard_space: str
    iterations: int
    limit_state_error: Optional[float]
    direction_error: Optional[float]

    _arrays: ClassVar[tuple] = ("design_point_x", "design_point_u", "alpha")

    @classmethod
    def from_analysis(cls, analysis: "_pystra.FORM") -> "FORMResult":
        """Copy a completed solver's numerical results and diagnostics."""
        valid = bool(analysis._converged and analysis._results_valid)
        standard_space = getattr(analysis.transform, "standard_space", "normal")
        if valid:
            beta = (
                float(analysis._beta)
                if standard_space == "normal"
                else analysis._get_equivalent_beta()
            )
        return cls(
            method="FORM",
            status="converged" if valid else "not_converged",
            message="Converged" if valid else "FORM iteration limit reached",
            n_limit_state_evaluations=analysis._n_evaluations,
            variable_names=tuple(analysis.model.get_variables()),
            failure_probability=float(analysis._Pf) if valid else None,
            beta=beta if valid else None,
            design_index=float(analysis._beta) if valid else None,
            design_point_x=(np.ravel(analysis._design_point_x()) if valid else None),
            design_point_u=np.ravel(analysis._u) if valid else None,
            alpha=np.ravel(analysis._alpha) if valid else None,
            standard_space=standard_space,
            iterations=analysis._i or 0,
            limit_state_error=analysis._e1,
            direction_error=analysis._e2,
            options=analysis.options,
        )

    def _rows(self):
        rows = super()._rows()
        if self.standard_space != "normal":
            rows.append(("Design index", self.design_index))
        return [*rows, ("Iterations", self.iterations)]

    def to_dataframe(self) -> DataFrame:
        """Return the design point and direction as a table by variable.

        Returns
        -------
        pandas.DataFrame
            Columns ``design_point_x``, ``design_point_u`` and ``alpha``,
            indexed by variable name.
        """
        import pandas as pd

        if self.design_point_x is None:
            raise ValueError("An unconverged FORM result has no design point")
        return pd.DataFrame(
            {
                "design_point_x": self.design_point_x,
                "design_point_u": self.design_point_u,
                "alpha": self.alpha,
            },
            index=pd.Index(self.variable_names, name="variable"),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class SORMResult(_ProbabilityResult):
    """Record of a SORM analysis.

    ``failure_probability`` and ``beta`` come from ``formula``, Breitung's
    asymptotic formula. ``approximations`` gives each formula's failure
    probability by name: ``"breitung"`` and ``"modified_breitung"``
    (Hohenbichler and Rackwitz's modification). An entry is ``None`` when the
    fitted curvatures leave that formula undefined; if the primary formula is
    undefined, the status is ``"not_converged"`` and there is no estimate.

    Attributes
    ----------
    form : FORMResult
        The FORM design point that the fit is built on.
    fit : {"curve", "point"}
        Curve fitting through the Hessian, or point fitting on the surface.
    curvatures : ndarray
        Curve fitting: the sorted principal curvatures, shape ``(n - 1,)``.
        Point fitting: the curvatures on the negative and positive side of
        each principal axis, as rows of shape ``(2, n - 1)``.
    formula : str
        The approximation reported as the estimate.
    approximations : Mapping[str, float or None]
        Failure probability from each formula.
    """

    form: FORMResult
    fit: str
    curvatures: np.ndarray
    formula: str
    approximations: Mapping

    _arrays: ClassVar[tuple] = ("curvatures",)
    _read_only: ClassVar[tuple] = ("approximations",)

    def _rows(self):
        return [
            *super()._rows(),
            ("Fit", self.fit),
            ("FORM reliability index", self.form.beta),
            *(
                (f"Pf, {name.replace('_', ' ')}", value)
                for name, value in self.approximations.items()
            ),
        ]


@dataclass(frozen=True, kw_only=True, eq=False)
class SimulationResult(_ProbabilityResult):
    """Record of a simulation estimate.

    Returned by crude Monte Carlo, importance sampling, line sampling and
    subset simulation.

    Attributes
    ----------
    coefficient_of_variation : float
        Estimated coefficient of variation of the failure probability;
        infinite when no failure was observed. Subset simulation's value
        ignores correlation between Markov-chain samples, so it is a lower
        bound.
    n_samples : int
        Samples used: points for crude Monte Carlo and importance sampling,
        lines for line sampling, and the total over all levels for subset
        simulation.
    diagnostics : Mapping
        Method-specific details. Crude Monte Carlo and importance sampling
        give ``history``: ``n_samples``, ``failure_probability`` and
        ``coefficient_of_variation`` after each block. Importance and line
        sampling give ``form``, the FORM record they sample about; line
        sampling adds ``direction``.
        Subset simulation gives ``thresholds``, ``conditional_probabilities``,
        ``n_levels`` and ``samples_per_level``.
    """

    coefficient_of_variation: float
    n_samples: int
    diagnostics: Mapping = field(default_factory=dict, repr=False)

    _read_only: ClassVar[tuple] = ("diagnostics",)

    def _rows(self):
        return [
            *super()._rows(),
            ("Coefficient of variation", self.coefficient_of_variation),
            ("Samples", self.n_samples),
        ]


@dataclass(frozen=True, kw_only=True, eq=False)
class SystemFORMResult(_ProbabilityResult):
    """Record of a system FORM approximation.

    Attributes
    ----------
    bounds : tuple of float
        Lower and upper bounds on the linearized event: Ditlevsen bounds for
        a series system, marginal bounds for a parallel system.
    component_results : Mapping[str, FORMResult]
        Each component's FORM record, by component name.
    correlation : ndarray
        Correlation of the linearized normal scores, ``alpha @ alpha.T``.
    intersections : ndarray
        Pairwise failure probabilities of the component tangent planes.
    """

    bounds: tuple
    component_results: Mapping
    correlation: np.ndarray
    intersections: np.ndarray

    _arrays: ClassVar[tuple] = ("correlation", "intersections")
    _read_only: ClassVar[tuple] = ("bounds", "component_results")

    def _rows(self):
        return [
            *super()._rows(),
            (
                "Bounds",
                (
                    "unavailable"
                    if self.bounds is None
                    else f"[{_text(self.bounds[0])}, {_text(self.bounds[1])}]"
                ),
            ),
            ("Components", len(self.component_results)),
        ]


@dataclass(frozen=True, kw_only=True, eq=False)
class SensitivityResult(_ProbabilityResult):
    """Record of a sensitivity analysis of the FORM reliability index.

    ``failure_probability`` and ``beta`` are those of ``form``, the FORM
    analysis at the model's parameters. The sensitivities are derivatives of
    its design index.

    Attributes
    ----------
    form : FORMResult
        FORM at the model's parameters.
    approach : {"numerical", "closed_form"}
        Forward finite differences, or Bourinet's (2017) closed form.
    marginal : Mapping[str, Mapping[str, float]]
        Derivative of the index with respect to each distribution parameter,
        by variable and parameter name.
    correlation : ndarray or None
        Closed form only: derivatives with respect to the correlation
        coefficients, symmetric with a zero diagonal.
    delta : float or None
        Numerical only: the relative perturbation.
    diagnostics : Mapping
        On nonconvergence, ``failed_form`` retains the failed inner FORM
        record and ``phase`` identifies the baseline or perturbation. A
        perturbation also records its variable, parameter and actual step.
    """

    form: FORMResult
    approach: str
    marginal: Mapping
    correlation: Optional[np.ndarray]
    delta: Optional[float]
    diagnostics: Mapping = field(default_factory=dict, repr=False)

    _arrays: ClassVar[tuple] = ("correlation",)
    _read_only: ClassVar[tuple] = ("marginal", "diagnostics")

    def _rows(self):
        return [*super()._rows(), ("Approach", self.approach)]

    def to_dataframe(self) -> DataFrame:
        """Return the marginal sensitivities as a tidy table.

        Returns
        -------
        pandas.DataFrame
            Columns ``Variable``, ``Parameter`` and ``∂β/∂θ``.
        """
        import pandas as pd

        return pd.DataFrame(
            [
                {"Variable": name, "Parameter": parameter, "∂β/∂θ": value}
                for name, parameters in self.marginal.items()
                for parameter, value in parameters.items()
            ]
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class StrongMaximumResult(_Result):
    """Record of a Strong Maximum Test, a diagnostic of a FORM design point.

    It is not a probability estimate, and finding no competing region does
    not certify the design point.

    Attributes
    ----------
    has_competing_points : bool
        Whether the far failure region contains sample points.
    design_point_u : ndarray
        The candidate design point in standard coordinates.
    design_index : float
        Its distance from the origin.
    radius : float
        Radius of the test sphere.
    point_number : int
        Points sampled on the sphere.
    confidence_level : float
        Nominal probability of detecting the reference cap.
    cap_probability : float
        Probability that one point falls in the reference cap.
    points_u, points_x : ndarray
        The sphere points in standard and physical coordinates, one per row.
    limit_state_values : ndarray
        Limit-state value at each point.
    regions : Mapping[str, ndarray]
        Boolean masks over the points: ``far_failure``, ``near_failure``,
        ``far_safe`` and ``near_safe``.
    """

    has_competing_points: bool
    design_point_u: np.ndarray
    design_index: float
    radius: float
    point_number: int
    confidence_level: float
    cap_probability: float
    points_u: np.ndarray = field(repr=False)
    points_x: np.ndarray = field(repr=False)
    limit_state_values: np.ndarray = field(repr=False)
    regions: Mapping = field(repr=False)

    _arrays: ClassVar[tuple] = (
        "design_point_u",
        "points_u",
        "points_x",
        "limit_state_values",
    )
    _read_only: ClassVar[tuple] = ("regions",)

    def points(self, region: str = "far_failure", space: str = "u") -> np.ndarray:
        """Return the points of one region as rows.

        Parameters
        ----------
        region : str, default "far_failure"
            ``far_failure``, ``near_failure``, ``far_safe`` or ``near_safe``.
        space : {"u", "x"}, default "u"
            Standard or physical coordinates.
        """
        if space not in ("u", "x"):
            raise ValueError("space must be 'u' or 'x'")
        return (self.points_u if space == "u" else self.points_x)[self.regions[region]]

    def _rows(self):
        return [
            ("Competing points", self.has_competing_points),
            ("Sphere points", self.point_number),
            ("Nominal confidence", self.confidence_level),
        ]


@dataclass(frozen=True, kw_only=True, eq=False)
class DistributionAnalysisResult(_Result):
    """Record of a distribution analysis: Monte Carlo samples of the model.

    Attributes
    ----------
    n_samples : int
        Samples drawn.
    bins : int
        Suggested number of histogram bins.
    samples_x : ndarray
        The samples in physical coordinates, shape ``(n_samples, n)``.
    limit_state_values : ndarray
        Limit-state value at each sample.
    """

    n_samples: int
    bins: int
    samples_x: np.ndarray = field(repr=False)
    limit_state_values: np.ndarray = field(repr=False)

    _arrays: ClassVar[tuple] = ("samples_x", "limit_state_values")

    def _rows(self):
        return [("Samples", self.n_samples), ("Histogram bins", self.bins)]
