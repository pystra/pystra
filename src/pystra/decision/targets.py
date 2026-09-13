"""Source-table and derived LQI reliability targets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from .swtp import SWTP, SWTP_TARGET_SOURCE

__all__ = [
    "TargetReliability",
    "lqi_k1",
    "lognormal_ratio_failure_probability",
    "TargetReliabilityCalibration",
    "RackwitzTargetModel",
    "derive_lqi_target",
    "rackwitz_table",
    "lqi_target_reliability",
]


@dataclass(frozen=True)
class TargetReliability:
    """Target failure probability and reliability index from a calibration.

    A single result type for every target-reliability route in this module: the
    rounded LQI table lookup (:meth:`LQI.lookup_target`), the LQI marginal
    optimization (:meth:`LQI.derive_target`), and the Rackwitz/Steenbergen
    code-calibration model (:meth:`RackwitzTargetModel.calibrate`).  Only ``pf``,
    ``beta`` and ``method`` are always populated; fields that do not apply to a
    given route are left as ``None``.

    Parameters
    ----------
    pf : float
        Annual failure probability or rate at the target.
    beta : float
        Reliability index corresponding to ``pf``.
    method : str
        Route that produced the target, e.g. ``"lqi-table"``,
        ``"LQI marginal"`` or ``"Rackwitz/Steenbergen"``.
    k1 : float, optional
        LQI safety cost ratio, when the target comes from the LQI route.
    cost_class : str, optional
        Discrete cost-class label from the rounded LQI table.
    variability : str, optional
        Variability class used for the rounded LQI table.
    design : float, optional
        Optimizing design parameter for calculated targets (the mean
        resistance-to-load ratio in the Rackwitz examples).
    objective : float, optional
        Objective value at the optimizing design.
    converged : bool, optional
        Whether the underlying optimization converged.
    message : str, optional
        Solver message for calculated targets.
    source : str, optional
        Literature source carried with a looked-up target.
    metadata : mapping, optional
        Additional model parameters or labels.
    """

    pf: float
    beta: float
    method: str
    k1: float | None = None
    cost_class: str | None = None
    variability: str | None = None
    design: float | None = None
    objective: float | None = None
    converged: bool | None = None
    message: str = ""
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.pf <= 1.0:
            raise ValueError("pf must be in [0, 1]")

    def to_dict(self) -> dict:
        """Return the populated scalar quantities as a dictionary."""

        data = {"method": self.method, "pf": self.pf, "beta": self.beta}
        for name in (
            "design",
            "objective",
            "k1",
            "cost_class",
            "variability",
            "converged",
            "source",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        data.update(self.metadata or {})
        return data

    def for_period(
        self, years: float, dependence_interval: float = 1.0
    ) -> "TargetReliability":
        """Return the target converted to a reference period of ``years``.

        The annual failure probability ``pf`` is compounded over ``years``,
        assuming the governing maximum renews every ``dependence_interval``
        years.  ``dependence_interval=1`` treats the annual maxima as
        independent (the most onerous case), while
        ``dependence_interval=years`` treats the period as fully dependent (a
        single renewal) and leaves the annual ``pf`` unchanged.

        The conversion uses ``pf`` directly rather than re-deriving an annual
        probability from ``beta``; for rounded table targets the two are not an
        exact pair, and compounding ``pf`` is the correct quantity.

        Parameters
        ----------
        years : float
            Reference period in years.
        dependence_interval : float, optional
            Renewal interval of the governing maximum, in years; must lie in
            ``(0, years]``.

        Returns
        -------
        TargetReliability
            A new target at the reference period, with the original ``method``
            and class labels retained and ``reference_period_years`` /
            ``dependence_interval`` recorded in ``metadata``.
        """

        if years <= 0:
            raise ValueError("years must be positive")
        if not 0 < dependence_interval <= years:
            raise ValueError("dependence_interval must be in (0, years]")

        annual_pf = float(self.pf)
        periods = years / dependence_interval
        period_pf = 1.0 - (1.0 - annual_pf) ** periods
        metadata = dict(self.metadata or {})
        metadata["reference_period_years"] = years
        metadata["dependence_interval"] = dependence_interval
        return replace(
            self,
            pf=period_pf,
            beta=_beta_from_failure_probability(period_pf),
            metadata=metadata,
        )


_TARGET_TABLE = (
    ("large", 1e-3, 1e-2, 1e-3, 3.1),
    ("medium", 1e-4, 1e-3, 1e-4, 3.7),
    ("small", 1e-5, 1e-4, 1e-5, 4.2),
)


_VARIABILITY_FACTORS = {"medium": 1.0, "high": 5.0, "low": 0.5}


def lqi_k1(
    safety_cost_rate: float,
    swtp: float | SWTP,
    expected_fatalities_given_failure: float,
) -> float:
    """Return the LQI safety cost ratio ``K1``.

    The Fischer, Barnardo, and Faber table uses a ratio of the marginal
    safety cost to the SWTP-weighted expected fatalities.  In their bridge
    notation the numerator is typically ``C1 * (gamma_s + omega)``.
    """

    value_per_life = swtp.value_per_life if isinstance(swtp, SWTP) else float(swtp)
    if safety_cost_rate <= 0:
        raise ValueError("safety_cost_rate must be positive")
    if value_per_life <= 0:
        raise ValueError("swtp must be positive")
    if expected_fatalities_given_failure <= 0:
        raise ValueError("expected_fatalities_given_failure must be positive")
    return safety_cost_rate / (value_per_life * expected_fatalities_given_failure)


def _as_scalar_or_array(value):
    array = np.asarray(value)
    return float(array) if array.ndim == 0 else array


def _beta_from_failure_probability(pf: float) -> float:
    if pf < 0 or pf > 1:
        raise ValueError("failure probability must be in [0, 1]")
    if pf == 0:
        return float("inf")
    if pf == 1:
        return float("-inf")
    return -float(norm.ppf(pf))


def lognormal_ratio_failure_probability(
    design: float | np.ndarray,
    resistance_cov: float,
    load_cov: float,
) -> float | np.ndarray:
    """Return ``P(R - S <= 0)`` for independent lognormal resistance and load.

    ``design`` is the ratio ``E[R] / E[S]``.  The expression is the closed-form
    reliability model used in the Rackwitz/JCSS target-reliability examples.
    """

    if resistance_cov < 0:
        raise ValueError("resistance_cov must be non-negative")
    if load_cov < 0:
        raise ValueError("load_cov must be non-negative")
    if resistance_cov == 0 and load_cov == 0:
        raise ValueError("at least one coefficient of variation must be positive")

    design_ratio = np.asarray(design, dtype=float)
    if np.any(design_ratio <= 0):
        raise ValueError("design must be positive")

    numerator = np.log(
        design_ratio * np.sqrt((1.0 + load_cov**2) / (1.0 + resistance_cov**2))
    )
    denominator = np.sqrt(np.log((1.0 + resistance_cov**2) * (1.0 + load_cov**2)))
    return _as_scalar_or_array(norm.cdf(-numerator / denominator))


@dataclass
class TargetReliabilityCalibration:
    """One-dimensional calibration for deriving a target reliability.

    The calibration maximizes a user-supplied objective over a scalar design
    variable and then evaluates the associated failure probability.  It is the
    generic calculation layer behind the built-in Rackwitz and LQI marginal
    target helpers.
    """

    objective: Callable[[float], float]
    failure_probability: Callable[[float], float]
    bounds: tuple[float, float] = (1.0, 15.0)
    variable: str = "design"
    metadata: Mapping[str, Any] | None = None

    def run(self) -> TargetReliability:
        """Run the bounded scalar optimization."""

        lower, upper = map(float, self.bounds)
        if lower >= upper:
            raise ValueError("bounds must be an increasing (lower, upper) pair")

        solution = minimize_scalar(
            lambda value: -float(self.objective(value)),
            bounds=(lower, upper),
            method="bounded",
        )
        design = float(solution.x)
        objective = float(self.objective(design))
        pf = float(self.failure_probability(design))
        beta = _beta_from_failure_probability(pf)

        metadata = dict(self.metadata or {})
        metadata.setdefault("variable", self.variable)

        return TargetReliability(
            pf=pf,
            beta=beta,
            method=str(metadata.get("method", "calibration")),
            design=design,
            objective=objective,
            converged=bool(solution.success),
            message=str(solution.message),
            metadata=metadata,
        )


@dataclass(frozen=True)
class RackwitzTargetModel:
    """Rackwitz/Steenbergen target-reliability model for code calibration.

    The model follows the normalized life-cycle objective used by Rackwitz and
    restated by Steenbergen, Rózsás, and Vrouwenvelder.  Every cost is expressed
    as a fraction of the base construction cost ``C0`` (``base_cost``), and the
    design variable is the mean resistance-to-load ratio ``p = E[R] / E[S]``.
    The construction cost is ``C(p) = C0 + C1 p = base_cost * (1 +
    safety_cost_ratio * p)``, so the cost inputs below are ratios to ``C0``.

    Parameters
    ----------
    safety_cost_ratio : float
        Marginal safety cost ``C1 / C0`` -- the extra construction cost per unit
        of ``p`` as a fraction of the base cost.  Larger values make safety
        relatively more expensive and lower the optimal target.
    failure_cost_ratio : float
        Failure (ULS) consequence cost ``H / C0``.
    resistance_cov, load_cov : float, optional
        Coefficients of variation of resistance and load in the closed-form
        lognormal ``P_f(p)`` model.
    base_cost : float, optional
        Base construction cost ``C0``.  Because every other cost is a ratio to
        it, its value does not change the calibrated target; defaults to 1.
    interest_rate : float, optional
        Discount/interest rate ``gamma``.
    obsolescence_rate : float, optional
        Obsolescence rate ``omega``.
    load_occurrence_rate : float, optional
        Load occurrence rate ``lambda`` (renewals per year).
    serviceability_cost_ratio : float, optional
        Serviceability (SLS) cost ``U / C0``.
    demolition_cost_ratio : float, optional
        Demolition/obsolescence cost ``A / C0``.
    serviceability_resistance_ratio : float, optional
        Ratio of the ULS to SLS resistance thresholds; the SLS check uses
        ``p / serviceability_resistance_ratio``.
    benefit_rate : float, optional
        Constant annual benefit ``b / C0``.  It is independent of ``p`` and so
        does not affect the optimum; defaults to 0.

    Notes
    -----
    To recalibrate a whole table for your own classes, pass ``safety_costs``
    (the ``C1 / C0`` values) and ``failure_costs`` (the ``H / C0`` values) to
    :meth:`table`.
    """

    safety_cost_ratio: float
    failure_cost_ratio: float
    resistance_cov: float = 0.3
    load_cov: float = 0.3
    base_cost: float = 1.0
    interest_rate: float = 0.035
    obsolescence_rate: float = 0.02
    load_occurrence_rate: float = 1.0
    serviceability_cost_ratio: float = 0.3
    demolition_cost_ratio: float = 0.2
    serviceability_resistance_ratio: float = 1.5
    benefit_rate: float = 0.0

    def __post_init__(self):
        if self.safety_cost_ratio <= 0:
            raise ValueError("safety_cost_ratio must be positive")
        if self.failure_cost_ratio < 0:
            raise ValueError("failure_cost_ratio must be non-negative")
        if self.base_cost <= 0:
            raise ValueError("base_cost must be positive")
        if self.interest_rate <= 0:
            raise ValueError("interest_rate must be positive")
        if self.obsolescence_rate < 0:
            raise ValueError("obsolescence_rate must be non-negative")
        if self.load_occurrence_rate < 0:
            raise ValueError("load_occurrence_rate must be non-negative")
        if self.serviceability_cost_ratio < 0:
            raise ValueError("serviceability_cost_ratio must be non-negative")
        if self.demolition_cost_ratio < 0:
            raise ValueError("demolition_cost_ratio must be non-negative")
        if self.serviceability_resistance_ratio <= 0:
            raise ValueError("serviceability_resistance_ratio must be positive")

    @property
    def metadata(self) -> dict:
        """Return source parameters used by the normalized model."""

        return {
            "method": "Rackwitz/Steenbergen",
            "safety_cost_ratio": self.safety_cost_ratio,
            "failure_cost_ratio": self.failure_cost_ratio,
            "resistance_cov": self.resistance_cov,
            "load_cov": self.load_cov,
            "base_cost": self.base_cost,
            "interest_rate": self.interest_rate,
            "obsolescence_rate": self.obsolescence_rate,
            "load_occurrence_rate": self.load_occurrence_rate,
            "serviceability_cost_ratio": self.serviceability_cost_ratio,
            "demolition_cost_ratio": self.demolition_cost_ratio,
            "serviceability_resistance_ratio": self.serviceability_resistance_ratio,
            "benefit_rate": self.benefit_rate,
        }

    def construction_cost(self, design: float) -> float:
        """Return ``C(p) = C0 + C1 p``."""

        if design <= 0:
            raise ValueError("design must be positive")
        return self.base_cost * (1.0 + self.safety_cost_ratio * float(design))

    def failure_probability(self, design: float) -> float:
        """Return the ULS failure probability for a design value."""

        return float(
            lognormal_ratio_failure_probability(
                design, self.resistance_cov, self.load_cov
            )
        )

    def serviceability_probability(self, design: float) -> float:
        """Return the SLS failure probability for a design value."""

        return float(
            lognormal_ratio_failure_probability(
                float(design) / self.serviceability_resistance_ratio,
                self.resistance_cov,
                self.load_cov,
            )
        )

    def objective(self, design: float) -> float:
        """Return the normalized Rackwitz/Steenbergen life-cycle objective."""

        construction = self.construction_cost(design)
        serviceability = self.base_cost * self.serviceability_cost_ratio
        demolition = self.base_cost * self.demolition_cost_ratio
        failure_cost = self.base_cost * self.failure_cost_ratio

        gamma = self.interest_rate
        benefit = self.base_cost * self.benefit_rate / gamma
        serviceability_loss = (
            serviceability
            * self.load_occurrence_rate
            / gamma
            * self.serviceability_probability(design)
        )
        obsolescence_loss = (construction + demolition) * self.obsolescence_rate / gamma
        failure_loss = (
            (construction + failure_cost)
            * self.load_occurrence_rate
            / gamma
            * self.failure_probability(design)
        )
        return float(
            benefit
            - construction
            - serviceability_loss
            - obsolescence_loss
            - failure_loss
        )

    def calibration(
        self, bounds: tuple[float, float] = (1.0, 15.0)
    ) -> TargetReliabilityCalibration:
        """Return a generic calibration object for this model."""

        return TargetReliabilityCalibration(
            objective=self.objective,
            failure_probability=self.failure_probability,
            bounds=bounds,
            variable="p",
            metadata=self.metadata,
        )

    def calibrate(self, bounds: tuple[float, float] = (1.0, 15.0)) -> TargetReliability:
        """Return the target reliability implied by this model."""

        return self.calibration(bounds=bounds).run()

    @classmethod
    def table(
        cls,
        safety_costs: Mapping[str, float] | None = None,
        failure_costs: Mapping[str, float] | None = None,
        bounds: tuple[float, float] = (1.0, 15.0),
        **kwargs,
    ) -> pd.DataFrame:
        """Calculate a Rackwitz-style target-reliability table.

        Parameters
        ----------
        safety_costs : mapping, optional
            Label -> ``C1 / C0`` (marginal safety cost) for each relative
            safety-cost class.  Defaults to representative ``large``/``normal``/
            ``small`` values.
        failure_costs : mapping, optional
            Label -> ``H / C0`` (failure consequence cost) for each consequence
            class.  Defaults to representative ``minor``/``moderate``/``large``
            values.
        bounds : tuple of float, optional
            Search interval for the optimizing ``p``.
        **kwargs
            Forwarded to :class:`RackwitzTargetModel` (e.g. ``resistance_cov``,
            ``interest_rate``) so every model setting can be varied too.

        Notes
        -----
        The defaults are representative values from the broad classes used in
        the literature, not a retyping of any rounded target table.  Supply your
        own ``safety_costs`` and ``failure_costs`` to recalibrate for different
        cost and consequence assumptions.
        """

        if safety_costs is None:
            safety_costs = {"large": 0.3, "normal": 0.03, "small": 0.003}
        if failure_costs is None:
            failure_costs = {"minor": 0.5, "moderate": 2.5, "large": 6.5}

        rows = []
        for safety_label, safety_cost_ratio in safety_costs.items():
            for failure_label, failure_cost_ratio in failure_costs.items():
                model = cls(
                    safety_cost_ratio=safety_cost_ratio,
                    failure_cost_ratio=failure_cost_ratio,
                    **kwargs,
                )
                result = model.calibrate(bounds=bounds)
                row = result.to_dict()
                row.update(
                    {
                        "relative_safety_cost": safety_label,
                        "failure_consequence": failure_label,
                    }
                )
                rows.append(row)

        columns = [
            "relative_safety_cost",
            "failure_consequence",
            "safety_cost_ratio",
            "failure_cost_ratio",
            "design",
            "pf",
            "beta",
            "objective",
            "resistance_cov",
            "load_cov",
            "base_cost",
            "interest_rate",
            "obsolescence_rate",
            "load_occurrence_rate",
            "serviceability_cost_ratio",
            "demolition_cost_ratio",
            "serviceability_resistance_ratio",
            "benefit_rate",
            "converged",
        ]
        return pd.DataFrame(rows, columns=columns)


def derive_lqi_target(
    k1: float,
    resistance_cov: float = 0.4,
    load_cov: float = 0.4,
    bounds: tuple[float, float] = (1.0, 20.0),
) -> TargetReliability:
    """Calculate an LQI target from the marginal optimization condition.

    This derives a target reliability by minimizing the non-dimensional
    expression ``K1 * p + P_f(p)`` for the same lognormal resistance-demand
    model used in the Rackwitz examples.  The table returned by
    :func:`lqi_target_reliability` remains the rounded source-table lookup.
    """

    if k1 <= 0:
        raise ValueError("k1 must be positive")

    def failure_probability(design):
        return lognormal_ratio_failure_probability(design, resistance_cov, load_cov)

    calibration = TargetReliabilityCalibration(
        objective=lambda design: -(k1 * design + failure_probability(design)),
        failure_probability=failure_probability,
        bounds=bounds,
        variable="p",
        metadata={
            "method": "LQI marginal",
            "k1": k1,
            "resistance_cov": resistance_cov,
            "load_cov": load_cov,
        },
    )
    return calibration.run()


def rackwitz_table(**kwargs) -> pd.DataFrame:
    """Calculate the default Rackwitz/Steenbergen target-reliability table."""

    return RackwitzTargetModel.table(**kwargs)


def lqi_target_reliability(k1: float, variability: str = "medium") -> TargetReliability:
    """Return an LQI target reliability for a safety cost ratio.

    Parameters
    ----------
    k1 : float
        Safety cost ratio.  Medium-variability classes follow the ranges in
        Fischer, Barnardo, and Faber (2012), Table 3.
    variability : {"low", "medium", "high"}, optional
        Approximate consequence/resistance variability adjustment.  The
        source table reports that high variability gives failure
        probabilities about five times larger and low variability about two
        times smaller than the medium case.
    """

    if k1 <= 0:
        raise ValueError("k1 must be positive")

    key = str(variability).strip().lower()
    if key not in _VARIABILITY_FACTORS:
        valid = ", ".join(sorted(_VARIABILITY_FACTORS))
        raise ValueError(f"variability must be one of: {valid}")
    factor = _VARIABILITY_FACTORS[key]

    cost_class = "extrapolated"
    pf = k1 / 5.0
    beta = -float(norm.ppf(pf))

    for name, lower, upper, table_pf, table_beta in _TARGET_TABLE:
        if lower <= k1 <= upper:
            cost_class = name
            pf = table_pf
            beta = table_beta
            break

    if factor != 1.0:
        pf = pf * factor
        pf = float(np.clip(pf, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
        beta = -float(norm.ppf(pf))

    return TargetReliability(
        pf=float(pf),
        beta=float(beta),
        method="lqi-table",
        k1=k1,
        cost_class=cost_class,
        variability=key,
        source=SWTP_TARGET_SOURCE,
    )
