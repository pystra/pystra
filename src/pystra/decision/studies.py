"""Reliability and risk evaluations over design alternatives."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd
from ..assessment import ReliabilityEstimate, _snapshot_result
from ..errors import AnalysisError
from ..reporting import reliability_row

from .risk import RiskResult, ScenarioRiskModel
from .swtp import SWTP

__all__ = ["DesignStudy", "DesignResult", "DesignStudyResult", "RiskStudy"]


def _coerce_analysis_result(result: Any) -> Mapping[str, Any]:
    return reliability_row(result, probability_name="pf")


@dataclass(frozen=True)
class DesignResult:
    """One design value and its original reliability result snapshot."""

    value: Any
    reliability: Any


@dataclass(frozen=True)
class DesignStudyResult:
    """Every design alternative, including failed analyses and diagnostics."""

    variable: str
    cases: tuple[DesignResult, ...]

    @property
    def converged(self) -> bool:
        """Whether every alternative met its evaluator's criteria."""
        return all(
            _coerce_analysis_result(case.reliability)["converged"]
            for case in self.cases
        )

    def to_frame(self, *, include_analysis: bool = False) -> pd.DataFrame:
        """Return a fresh table; failed estimates are NaN, with status retained."""
        rows = []
        for case in self.cases:
            row = {
                self.variable: deepcopy(case.value),
                **_coerce_analysis_result(case.reliability),
            }
            if include_analysis:
                row["analysis"] = _snapshot_result(case.reliability)
            rows.append(row)
        columns = [
            self.variable,
            "pf",
            "beta",
            "converged",
            "status",
            "message",
            "method",
        ]
        if include_analysis:
            columns.append("analysis")
        return pd.DataFrame(rows, columns=columns)


@dataclass
class DesignStudy:
    """Evaluate a reliability callback over a one-dimensional design range.

    Parameters
    ----------
    variable : str
        Design-variable name for output tables; must not shadow result columns.
    values : iterable
        Design alternatives, copied to a tuple so repeated runs retain the grid.
    analysis : callable
        Called once for each value. May return a reliability record, an analytic
        probability, a (pf, beta) pair, or a mapping with pf and/or beta. Records
        retain status and diagnostics. AnalysisError becomes a failed alternative;
        invalid specifications and programming errors still raise.
    """

    variable: str
    values: Iterable[Any]
    analysis: Callable[[Any], Any]

    def __post_init__(self) -> None:
        reserved = {
            "pf",
            "beta",
            "converged",
            "status",
            "message",
            "method",
            "analysis",
        }
        if (
            not isinstance(self.variable, str)
            or not self.variable
            or self.variable in reserved
        ):
            raise ValueError(
                "variable must be a nonempty name distinct from result columns"
            )
        if not callable(self.analysis):
            raise TypeError("analysis must be callable")
        self.values = tuple(deepcopy(tuple(self.values)))
        if not self.values:
            raise ValueError("A design study needs at least one alternative")

    def run(self) -> DesignStudyResult:
        """Return design/result snapshots for all alternatives, including failures."""
        records = []
        for value in self.values:
            try:
                result = self.analysis(deepcopy(value))
            except AnalysisError as error:
                result = error.result
                if result is None:
                    result = ReliabilityEstimate(
                        method="callback", status="not_converged", message=str(error)
                    )
                elif _coerce_analysis_result(result)["converged"]:
                    raise ValueError(
                        "AnalysisError must carry a failed result"
                    ) from error
            _coerce_analysis_result(result)
            records.append(DesignResult(deepcopy(value), _snapshot_result(result)))
        return DesignStudyResult(self.variable, tuple(records))

    def evaluate(self, include_analysis: bool = False) -> pd.DataFrame:
        """Run the callback and return estimates with explicit status columns."""
        return self.run().to_frame(include_analysis=include_analysis)


@dataclass
class RiskStudy:
    """Evaluate a risk model over design alternatives."""

    variable: str
    values: Iterable[Any]
    model: ScenarioRiskModel | Callable[[Any], RiskResult | Any]

    def _evaluate_model(self, value: Any) -> RiskResult:
        if isinstance(self.model, ScenarioRiskModel):
            return self.model.evaluate(value)

        result = self.model(value)
        if isinstance(result, RiskResult):
            return result
        return RiskResult.from_scenarios(result, metadata={self.variable: value})

    def evaluate(self, swtp: float | SWTP | None = None) -> pd.DataFrame:
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
