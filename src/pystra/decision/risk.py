"""Scenario consequences and annual risk quantities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from .swtp import SWTP, _swtp_value
from .targets import _as_scalar_or_array

__all__ = [
    "FatalityConsequence",
    "RiskResult",
    "ScenarioRiskModel",
    "jcss_lqi_risk_cost",
    "jcss_lqi_risk_cost_from_result",
]


@dataclass(frozen=True)
class FatalityConsequence:
    """Expected fatalities conditional on structural failure."""

    people_exposed: float
    probability_death_given_failure: float = 1.0

    def __post_init__(self):
        if self.people_exposed < 0:
            raise ValueError("people_exposed must be non-negative")
        if not 0 <= self.probability_death_given_failure <= 1:
            raise ValueError("probability_death_given_failure must be in [0, 1]")

    @property
    def expected_fatalities_given_failure(self) -> float:
        """Return expected fatalities conditional on failure."""

        return self.people_exposed * self.probability_death_given_failure


@dataclass(frozen=True)
class RiskResult:
    """Expected annual risk quantities for a component or scenario model.

    Parameters
    ----------
    annual_failure_rate : float
        Annual probability or rate of the represented failure event.  For
        scenario studies this may be the sum of joint failure-state rates.
    expected_fatalities : float
        Expected fatalities per year.
    expected_economic_loss : float
        Expected economic loss per year, excluding SWTP life-safety valuation.
    scenarios : pandas.DataFrame, optional
        Scenario table used to derive the expected values.  This is useful for
        joint failure states and nonlinear consequence models.
    metadata : mapping, optional
        User-supplied context such as model name, design point, or units.
    """

    annual_failure_rate: float
    expected_fatalities: float = 0.0
    expected_economic_loss: float = 0.0
    scenarios: pd.DataFrame | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        if self.annual_failure_rate < 0:
            raise ValueError("annual_failure_rate must be non-negative")
        if self.expected_fatalities < 0:
            raise ValueError("expected_fatalities must be non-negative")
        if self.expected_economic_loss < 0:
            raise ValueError("expected_economic_loss must be non-negative")
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @classmethod
    def from_scenarios(
        cls,
        scenarios: Any,
        weight_col: str = "probability",
        fatalities_col: str = "fatalities",
        economic_loss_col: str = "economic_loss",
        failure_col: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RiskResult":
        """Aggregate a scenario table into annual risk quantities.

        The scenario weights may be annual probabilities or annual rates.  No
        independence assumptions are made, so rows may represent joint
        failure states produced by an upstream reliability model.
        If ``failure_col`` is omitted, every row is treated as a failure/loss
        scenario for the purpose of ``annual_failure_rate``.
        """

        df = pd.DataFrame(scenarios).copy()
        for column in (weight_col, fatalities_col, economic_loss_col):
            if column not in df:
                raise KeyError(f"Scenario table is missing column {column!r}")
        if failure_col is not None and failure_col not in df:
            raise KeyError(f"Scenario table is missing column {failure_col!r}")

        weights = df[weight_col].astype(float)
        fatalities = df[fatalities_col].astype(float)
        economic_loss = df[economic_loss_col].astype(float)

        if (weights < 0).any():
            raise ValueError("scenario weights must be non-negative")
        if (fatalities < 0).any():
            raise ValueError("scenario fatalities must be non-negative")
        if (economic_loss < 0).any():
            raise ValueError("scenario economic losses must be non-negative")

        if failure_col is None:
            annual_failure_rate = float(weights.sum())
        else:
            annual_failure_rate = float((weights * df[failure_col].astype(bool)).sum())

        df["expected_fatalities_contribution"] = weights * fatalities
        df["expected_economic_loss_contribution"] = weights * economic_loss

        return cls(
            annual_failure_rate=annual_failure_rate,
            expected_fatalities=float(df["expected_fatalities_contribution"].sum()),
            expected_economic_loss=float(
                df["expected_economic_loss_contribution"].sum()
            ),
            scenarios=df,
            metadata=metadata,
        )

    @property
    def beta(self) -> float | None:
        """Return the generalized reliability index when rate is probability-like."""

        if self.annual_failure_rate == 0:
            return float("inf")
        if 0 < self.annual_failure_rate < 1:
            return -float(norm.ppf(self.annual_failure_rate))
        return None

    def get_failure(self) -> float:
        """Return ``annual_failure_rate`` for reliability-result compatibility."""

        return self.annual_failure_rate

    def get_beta(self) -> float | None:
        """Return ``beta`` for reliability-result compatibility."""

        return self.beta

    def life_safety_cost(self, swtp: float | SWTP) -> float:
        """Return SWTP-valued expected annual life-safety cost."""

        return _swtp_value(swtp) * self.expected_fatalities

    def total_risk_cost(
        self,
        swtp: float | SWTP | None = None,
        include_life_safety: bool = True,
    ) -> float:
        """Return expected annual economic plus optional life-safety risk cost."""

        total = self.expected_economic_loss
        if include_life_safety and swtp is not None:
            total += self.life_safety_cost(swtp)
        return float(total)

    def to_dict(self) -> dict:
        """Return scalar risk quantities as a dictionary."""

        data = {
            "annual_failure_rate": self.annual_failure_rate,
            "beta": self.beta,
            "expected_fatalities": self.expected_fatalities,
            "expected_economic_loss": self.expected_economic_loss,
        }
        data.update(self.metadata or {})
        return data


@dataclass
class ScenarioRiskModel:
    """Scenario-table risk model for aggregated risk studies.

    ``scenarios`` may be a dataframe-like object or a callable returning one.
    Callable scenarios are evaluated with the design value when supplied.
    """

    scenarios: Any | Callable[..., Any]
    weight_col: str = "probability"
    fatalities_col: str = "fatalities"
    economic_loss_col: str = "economic_loss"
    failure_col: str | None = None
    metadata: Mapping[str, Any] | None = None

    def evaluate(self, design: Any = None) -> RiskResult:
        """Evaluate the scenario model and return a :class:`RiskResult`."""

        if callable(self.scenarios):
            scenario_data = (
                self.scenarios(design) if design is not None else self.scenarios()
            )
        else:
            scenario_data = self.scenarios

        metadata = dict(self.metadata or {})
        if design is not None:
            metadata.setdefault("design", design)

        return RiskResult.from_scenarios(
            scenario_data,
            weight_col=self.weight_col,
            fatalities_col=self.fatalities_col,
            economic_loss_col=self.economic_loss_col,
            failure_col=self.failure_col,
            metadata=metadata,
        )


def jcss_lqi_risk_cost(
    safety_cost: float | np.ndarray,
    failure_rate: float | np.ndarray,
    swtp: float | SWTP,
    expected_fatalities_given_failure: float,
) -> float | np.ndarray:
    """Return the JCSS LQI life-safety risk cost.

    This implements the canonical LQI optimization term
    ``S(p) = C(p) + G_x k N_F h(p)`` from the JCSS background documents,
    where ``h(p)`` is a failure rate or annual failure probability.
    """

    value_per_life = _swtp_value(swtp)
    if value_per_life <= 0:
        raise ValueError("swtp must be positive")
    if expected_fatalities_given_failure < 0:
        raise ValueError("expected_fatalities_given_failure must be non-negative")
    result = np.asarray(
        safety_cost
    ) + value_per_life * expected_fatalities_given_failure * np.asarray(failure_rate)
    return _as_scalar_or_array(result)


def jcss_lqi_risk_cost_from_result(
    safety_cost: float,
    risk: RiskResult,
    swtp: float | SWTP,
    include_economic_loss: bool = False,
) -> float:
    """Return JCSS LQI risk cost from an aggregated risk result.

    This form is intended for scenario studies where the upstream model already
    provides expected annual fatalities.  Set
    ``include_economic_loss=True`` when the objective should include nonlinear
    economic consequences in addition to the SWTP life-safety term.
    """

    if safety_cost < 0:
        raise ValueError("safety_cost must be non-negative")
    risk_cost = risk.life_safety_cost(swtp)
    if include_economic_loss:
        risk_cost += risk.expected_economic_loss
    return float(safety_cost + risk_cost)
