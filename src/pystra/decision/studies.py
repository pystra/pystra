"""Reliability and risk evaluations over design alternatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd
from scipy.stats import norm

from .risk import RiskResult, ScenarioRiskModel
from .swtp import SWTP

__all__ = ["DesignStudy", "RiskStudy"]


def _coerce_analysis_result(result: Any) -> Mapping[str, float]:
    if isinstance(result, Mapping):
        pf = (
            result.get("pf")
            if "pf" in result
            else result.get("failure_probability", result.get("failure"))
        )
        beta = result.get("beta", result.get("reliability_index"))
    elif hasattr(result, "failure_probability") or hasattr(result, "beta"):
        pf = getattr(result, "failure_probability", None)
        beta = getattr(result, "beta", None)
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        pf = result[0] if len(result) > 0 else None
        beta = result[1] if len(result) > 1 else None
    else:
        pf = result
        beta = None

    if pf is None and beta is None:
        raise ValueError("analysis result must provide pf and/or beta")
    if pf is None:
        pf = float(norm.cdf(-float(beta)))
    if beta is None:
        beta = -float(norm.ppf(float(pf)))
    return {"pf": float(pf), "beta": float(beta)}


@dataclass
class DesignStudy:
    """Evaluate reliability results over a one-dimensional design range."""

    variable: str
    values: Iterable[Any]
    analysis: Callable[[Any], Any]

    def evaluate(self, include_analysis: bool = False) -> pd.DataFrame:
        """Run the analysis callback for each design value."""

        rows = []
        for value in self.values:
            result = self.analysis(value)
            data = dict(_coerce_analysis_result(result))
            data[self.variable] = value
            if include_analysis:
                data["analysis"] = result
            rows.append(data)
        columns = [self.variable, "pf", "beta"]
        if include_analysis:
            columns.append("analysis")
        return pd.DataFrame(rows, columns=columns)


@dataclass
class RiskStudy:
    """Evaluate a risk model over design alternatives."""

    variable: str
    values: Iterable[Any]
    model: Union[ScenarioRiskModel, Callable[[Any], Union[RiskResult, Any]]]

    def _evaluate_model(self, value: Any) -> RiskResult:
        if isinstance(self.model, ScenarioRiskModel):
            return self.model.evaluate(value)

        result = self.model(value)
        if isinstance(result, RiskResult):
            return result
        return RiskResult.from_scenarios(result, metadata={self.variable: value})

    def evaluate(self, swtp: Optional[Union[float, SWTP]] = None) -> pd.DataFrame:
        """Return risk quantities for each design value.

        The annual failure rate is also exposed as a ``pf`` column so that the
        same :class:`DDO` orchestration, objectives, and criteria used with a
        :class:`DesignStudy` accept a :class:`RiskStudy` unchanged.
        """

        rows = []
        for value in self.values:
            risk = self._evaluate_model(value)
            row = risk.to_dict()
            row["pf"] = risk.annual_failure_rate
            row[self.variable] = value
            if swtp is not None:
                row["life_safety_cost"] = risk.life_safety_cost(swtp)
                row["total_risk_cost"] = risk.total_risk_cost(swtp)
            rows.append(row)

        columns = [
            self.variable,
            "annual_failure_rate",
            "pf",
            "beta",
            "expected_fatalities",
            "expected_economic_loss",
        ]
        if swtp is not None:
            columns.extend(["life_safety_cost", "total_risk_cost"])
        extra_columns = [
            column for column in pd.DataFrame(rows).columns if column not in columns
        ]
        return pd.DataFrame(rows, columns=columns + extra_columns)
