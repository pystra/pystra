"""Economic objectives and annual cost conversions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "jcss_systematic_reconstruction_objective",
    "present_value_factor",
    "annualized_safety_cost",
    "DDOObjective",
    "CostBenefitModel",
]


def jcss_systematic_reconstruction_objective(
    benefit_rate: float,
    safety_cost: float,
    failure_rate: float,
    failure_consequence: float,
    discount_rate: float,
) -> float:
    """Return the JCSS infinite-horizon reconstruction objective.

    The expression follows the JCSS resistance-demand example:
    ``Z(p) = b / gamma - C(p) - (C(p) + H) h(p) / gamma``.
    """

    if discount_rate <= 0:
        raise ValueError("discount_rate must be positive")
    if not 0 <= failure_rate:
        raise ValueError("failure_rate must be non-negative")
    if safety_cost < 0:
        raise ValueError("safety_cost must be non-negative")
    if failure_consequence < 0:
        raise ValueError("failure_consequence must be non-negative")
    return float(
        benefit_rate / discount_rate
        - safety_cost
        - (safety_cost + failure_consequence) * failure_rate / discount_rate
    )


def present_value_factor(interest_rate: float, service_life: float) -> float:
    """Return the JCSS-style present value factor for a constant annual flow."""

    if service_life <= 0:
        raise ValueError("service_life must be positive")
    if interest_rate < 0:
        raise ValueError("interest_rate must be non-negative")
    if interest_rate == 0:
        return float(service_life)
    return float(
        (1.0 - (1.0 + interest_rate) ** (-service_life)) / np.log1p(interest_rate)
    )


def annualized_safety_cost(
    initial_cost: float,
    failure_probability: float,
    service_life: float,
    interest_rate: float,
    replacement_cost: float | None = None,
) -> float:
    """Return the annualized safety cost used by the JCSS LQI example."""

    if initial_cost < 0:
        raise ValueError("initial_cost must be non-negative")
    if not 0 <= failure_probability <= 1:
        raise ValueError("failure_probability must be in [0, 1]")
    if service_life <= 0:
        raise ValueError("service_life must be positive")
    if interest_rate < 0:
        raise ValueError("interest_rate must be non-negative")
    replacement = initial_cost if replacement_cost is None else replacement_cost
    if replacement < 0:
        raise ValueError("replacement_cost must be non-negative")

    if interest_rate == 0:
        adjustment = -1.0
    else:
        adjustment = (1.0 - (1.0 + interest_rate)) / np.log1p(interest_rate)
    return float(
        (initial_cost + replacement * adjustment * failure_probability) / service_life
    )


class DDOObjective:
    """Base interface for DDO objectives."""

    name = "objective"
    objective_column = "objective"

    def evaluate(self, results: pd.DataFrame, design: str) -> pd.DataFrame:
        """Return successful decision rows with objective-specific columns.

        Preserve the input row indices so DDO can align these values with
        failed alternatives, which remain visible with unavailable objectives.
        """

        raise NotImplementedError


@dataclass
class CostBenefitModel(DDOObjective):
    """Cost-benefit objective for a reliability design study."""

    benefit_rate: float
    interest_rate: float
    service_life: float
    construction_cost: float | Callable[[Any], float]
    failure_cost: float | Callable[[Any], float]

    def _value(self, item: float | Callable[[Any], float], design: Any) -> float:
        value = item(design) if callable(item) else item
        return float(value)

    def present_value_factor(self) -> float:
        """Return the present value factor for this model."""

        return present_value_factor(self.interest_rate, self.service_life)

    def construction(self, design: Any) -> float:
        """Return the construction or safety cost for a design value."""

        return self._value(self.construction_cost, design)

    def consequence(self, design: Any) -> float:
        """Return the failure consequence cost for a design value."""

        return self._value(self.failure_cost, design)

    def objective(self, design: Any, failure_probability: float) -> float:
        """Return the net present value objective for a design point."""

        if not 0 <= failure_probability <= 1:
            raise ValueError("failure_probability must be in [0, 1]")
        pvf = self.present_value_factor()
        construction = self.construction(design)
        failure_consequence = self.consequence(design)
        return float(
            self.benefit_rate * pvf
            - construction
            - failure_probability * failure_consequence * pvf
        )

    def annualized_safety_cost(
        self,
        design: Any,
        failure_probability: float,
        replacement_cost: float | None = None,
    ) -> float:
        """Return the JCSS annualized safety cost for a design point."""

        construction = self.construction(design)
        replacement = construction if replacement_cost is None else replacement_cost
        return annualized_safety_cost(
            construction,
            failure_probability,
            self.service_life,
            self.interest_rate,
            replacement,
        )

    def evaluate(self, results: pd.DataFrame, design: str) -> pd.DataFrame:
        """Return results with cost-benefit objective columns."""

        df = results.copy()
        if design not in df:
            raise KeyError(f"Design column {design!r} is missing")
        if "pf" not in df:
            raise KeyError("CostBenefitModel requires a 'pf' column")

        design_values = df[design].to_numpy()
        df[self.objective_column] = [
            self.objective(design_value, pf)
            for design_value, pf in zip(design_values, df["pf"])
        ]
        df["annualized_safety_cost"] = [
            self.annualized_safety_cost(design_value, pf)
            for design_value, pf in zip(design_values, df["pf"])
        ]
        return df
