"""Life-safety acceptability criteria for decision studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .risk import (
    FatalityConsequence,
    RiskResult,
    jcss_lqi_risk_cost,
    jcss_lqi_risk_cost_from_result,
)
from .swtp import SWTP, _require_explicit_indexed, _swtp_value
from .targets import (
    TargetReliability,
    derive_lqi_target,
    lqi_k1,
    lqi_target_reliability,
)

__all__ = [
    "jcss_lqi_acceptability_margin",
    "finite_difference_derivative",
    "jcss_lqi_acceptability",
    "jcss_lqi_is_acceptable",
    "DDOCriterion",
    "LQI",
]


def jcss_lqi_acceptability_margin(
    safety_cost_derivative: float,
    failure_rate_derivative: float,
    swtp: Union[float, SWTP],
    expected_fatalities_given_failure: float,
) -> float:
    """Return the JCSS marginal LQI acceptability margin.

    For a design parameter that increases safety, the JCSS condition is
    ``dC/dp >= -G_x k N_F dh/dp``.  The returned margin is therefore
    ``dC/dp + G_x k N_F dh/dp``; non-negative values satisfy the criterion.
    """

    value_per_life = _swtp_value(swtp)
    if value_per_life <= 0:
        raise ValueError("swtp must be positive")
    if expected_fatalities_given_failure < 0:
        raise ValueError("expected_fatalities_given_failure must be non-negative")
    return float(
        safety_cost_derivative
        + value_per_life * expected_fatalities_given_failure * failure_rate_derivative
    )


def finite_difference_derivative(
    function: Callable[[float], float],
    design: float,
    step: Optional[float] = None,
) -> float:
    """Return a central finite-difference derivative for a scalar design."""

    x = float(design)
    h = max(abs(x) * 1e-6, 1e-6) if step is None else float(step)
    if h <= 0:
        raise ValueError("step must be positive")
    return float((function(x + h) - function(x - h)) / (2.0 * h))


def jcss_lqi_acceptability(
    safety_cost: Callable[[float], float],
    failure_rate: Callable[[float], float],
    design: float,
    swtp: Union[float, SWTP],
    expected_fatalities_given_failure: float,
    step: Optional[float] = None,
) -> float:
    """Return the finite-difference JCSS LQI acceptability margin."""

    dcost = finite_difference_derivative(safety_cost, design, step=step)
    drate = finite_difference_derivative(failure_rate, design, step=step)
    return jcss_lqi_acceptability_margin(
        dcost, drate, swtp, expected_fatalities_given_failure
    )


def jcss_lqi_is_acceptable(
    safety_cost: Callable[[float], float],
    failure_rate: Callable[[float], float],
    design: float,
    swtp: Union[float, SWTP],
    expected_fatalities_given_failure: float,
    step: Optional[float] = None,
) -> bool:
    """Return ``True`` when the JCSS marginal LQI condition is satisfied."""

    margin = jcss_lqi_acceptability(
        safety_cost,
        failure_rate,
        design,
        swtp,
        expected_fatalities_given_failure,
        step=step,
    )
    return margin >= 0.0


class DDOCriterion:
    """Base interface for DDO acceptability criteria."""

    name = "criterion"
    feasibility_column: Optional[str] = None

    def evaluate(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return decision results with criterion-specific columns."""

        raise NotImplementedError

    def feasible(self, results: pd.DataFrame) -> pd.Series:
        """Return a boolean feasibility mask for evaluated results."""

        if self.feasibility_column is None:
            raise ValueError(f"{self.__class__.__name__} does not define feasibility")
        if self.feasibility_column not in results:
            raise KeyError(f"Feasibility column {self.feasibility_column!r} is missing")
        return results[self.feasibility_column].astype(bool)


def _lqi_consequence(
    expected_fatalities_given_failure: Optional[float],
    consequence: Optional[FatalityConsequence] = None,
) -> FatalityConsequence:
    if consequence is not None and expected_fatalities_given_failure is not None:
        raise ValueError(
            "Specify either expected_fatalities_given_failure or consequence, not both"
        )
    if consequence is not None:
        return consequence
    if expected_fatalities_given_failure is None:
        raise ValueError(
            "LQI construction requires expected_fatalities_given_failure or consequence"
        )
    return FatalityConsequence(people_exposed=expected_fatalities_given_failure)


@dataclass(frozen=True)
class LQI(DDOCriterion):
    """Minimum acceptable life-safety criterion using LQI/SWTP.

    The criterion adds consequence valuation and LQI acceptability columns to
    the reliability and objective results produced by :class:`DDO`.  It does
    not select the economic optimum by itself.
    """

    swtp: Optional[SWTP] = None
    consequence: Optional[FatalityConsequence] = None
    target: Optional[TargetReliability] = None

    name = "lqi"
    feasibility_column = "lqi_acceptable"

    @staticmethod
    def _as_swtp(swtp: Union[float, SWTP]) -> SWTP:
        return swtp if isinstance(swtp, SWTP) else SWTP(float(swtp))

    @staticmethod
    def lookup_target(k1: float, variability: str = "medium") -> TargetReliability:
        """Return the rounded source-table LQI target for ``k1``."""

        return lqi_target_reliability(k1, variability=variability)

    @staticmethod
    def derive_target(
        k1: float,
        resistance_cov: float = 0.4,
        load_cov: float = 0.4,
        bounds: tuple[float, float] = (1.0, 20.0),
    ) -> TargetReliability:
        """Calculate an LQI target from the marginal optimization condition."""

        return derive_lqi_target(
            k1,
            resistance_cov=resistance_cov,
            load_cov=load_cov,
            bounds=bounds,
        )

    @classmethod
    def from_swtp(
        cls,
        swtp: Union[float, SWTP],
        *,
        expected_fatalities_given_failure: Optional[float] = None,
        consequence: Optional[FatalityConsequence] = None,
        marginal_safety_cost: float,
        variability: str = "medium",
    ) -> "LQI":
        """Create an LQI criterion from an SWTP value.

        Parameters
        ----------
        swtp : float or SWTP
            Societal willingness to pay per statistical life.
        expected_fatalities_given_failure : float, optional
            Expected fatalities conditional on failure.
        consequence : FatalityConsequence, optional
            Consequence model.  Use this instead of
            ``expected_fatalities_given_failure`` when exposure and fatality
            probability should remain explicit.
        marginal_safety_cost : float
            Marginal annual safety cost used in the LQI target-reliability
            table.
        variability : {"low", "medium", "high"}, optional
            Variability class for the LQI target-reliability table.
        """

        swtp_value = cls._as_swtp(swtp)
        consequence = _lqi_consequence(expected_fatalities_given_failure, consequence)
        target = lqi_target_reliability(
            lqi_k1(
                safety_cost_rate=marginal_safety_cost,
                swtp=swtp_value,
                expected_fatalities_given_failure=(
                    consequence.expected_fatalities_given_failure
                ),
            ),
            variability=variability,
        )
        return cls(swtp=swtp_value, consequence=consequence, target=target)

    @classmethod
    def from_country(
        cls,
        code: str,
        *,
        expected_fatalities_given_failure: Optional[float] = None,
        consequence: Optional[FatalityConsequence] = None,
        marginal_safety_cost: float,
        indexed: Optional[bool] = None,
        variability: str = "medium",
    ) -> "LQI":
        """Create an LQI criterion from a built-in country SWTP value."""

        return cls.from_swtp(
            SWTP.from_country(code, indexed=_require_explicit_indexed(indexed)),
            expected_fatalities_given_failure=expected_fatalities_given_failure,
            consequence=consequence,
            marginal_safety_cost=marginal_safety_cost,
            variability=variability,
        )

    @classmethod
    def from_lqi(
        cls,
        *,
        gross_domestic_product_per_capita: float,
        work_leisure_parameter: float,
        demographic_constant: float,
        expected_fatalities_given_failure: Optional[float] = None,
        consequence: Optional[FatalityConsequence] = None,
        marginal_safety_cost: float,
        variability: str = "medium",
        currency: str = "currency units",
        price_year: Optional[int] = None,
        source: Optional[str] = "LQI relation SWTP = g / q * G",
    ) -> "LQI":
        """Create an LQI criterion from the LQI SWTP relation.

        ``work_leisure_parameter`` is the dimensionless LQI parameter ``q``
        (~0.1--0.2), not an annual mortality rate.
        """

        return cls.from_swtp(
            SWTP.from_lqi(
                gross_domestic_product_per_capita=gross_domestic_product_per_capita,
                work_leisure_parameter=work_leisure_parameter,
                demographic_constant=demographic_constant,
                currency=currency,
                price_year=price_year,
                source=source,
            ),
            expected_fatalities_given_failure=expected_fatalities_given_failure,
            consequence=consequence,
            marginal_safety_cost=marginal_safety_cost,
            variability=variability,
        )

    @property
    def expected_fatalities_given_failure(self) -> Optional[float]:
        """Return expected fatalities conditional on failure when available."""

        if self.consequence is None:
            return None
        return self.consequence.expected_fatalities_given_failure

    @property
    def k1(self) -> Optional[float]:
        """Return the LQI safety cost ratio when a target is available."""

        if self.target is None:
            return None
        return self.target.k1

    def _swtp_and_expected_fatalities(self) -> tuple[SWTP, float]:
        if self.swtp is None:
            raise ValueError("LQI criterion requires an SWTP value")
        if self.consequence is None:
            raise ValueError("LQI criterion requires a consequence model")
        return self.swtp, self.consequence.expected_fatalities_given_failure

    def risk_cost(
        self,
        safety_cost: Union[float, np.ndarray],
        failure_rate: Union[float, np.ndarray],
    ):
        """Return the JCSS LQI life-safety risk-cost term."""

        swtp, expected = self._swtp_and_expected_fatalities()
        return jcss_lqi_risk_cost(safety_cost, failure_rate, swtp, expected)

    def risk_cost_from_result(
        self,
        safety_cost: float,
        risk: RiskResult,
        include_economic_loss: bool = False,
    ) -> float:
        """Return LQI risk cost from an aggregated risk result."""

        swtp, _ = self._swtp_and_expected_fatalities()
        return jcss_lqi_risk_cost_from_result(
            safety_cost,
            risk,
            swtp,
            include_economic_loss=include_economic_loss,
        )

    def acceptability_margin(
        self,
        safety_cost_derivative: float,
        failure_rate_derivative: float,
    ) -> float:
        """Return the JCSS marginal LQI acceptability margin."""

        swtp, expected = self._swtp_and_expected_fatalities()
        return jcss_lqi_acceptability_margin(
            safety_cost_derivative,
            failure_rate_derivative,
            swtp,
            expected,
        )

    def acceptability_margin_at(
        self,
        safety_cost: Callable[[float], float],
        failure_rate: Callable[[float], float],
        design: float,
        step: Optional[float] = None,
    ) -> float:
        """Return the finite-difference LQI acceptability margin at a design.

        This is :meth:`acceptability_margin` evaluated from cost and failure-rate
        callables, differentiating each by central finite differences.
        """

        dcost = finite_difference_derivative(safety_cost, design, step=step)
        drate = finite_difference_derivative(failure_rate, design, step=step)
        return self.acceptability_margin(dcost, drate)

    def acceptability_boundary(
        self,
        safety_cost: Callable[[float], float],
        failure_rate: Callable[[float], float],
        bounds: tuple[float, float],
        step: Optional[float] = None,
    ) -> float:
        """Return the design where the LQI acceptability margin changes sign.

        The marginal acceptability margin ``dC/dp + SWTP * N_F * dh/dp`` is
        negative where society would still pay to reduce risk and non-negative
        once the marginal cost of safety meets or exceeds the SWTP-valued risk
        reduction.  The boundary is the design at which the margin is zero: the
        minimum design the LQI criterion accepts (the marginal acceptability
        boundary), which is a minimum safety requirement, not the economic
        optimum.  Root finding uses Brent's method over ``bounds`` and requires
        the margin to change sign across the interval.

        Parameters
        ----------
        safety_cost : callable
            Safety or construction cost as a function of the design value.
        failure_rate : callable
            Annual failure probability or rate as a function of the design.
        bounds : tuple of float
            Increasing ``(lower, upper)`` search interval bracketing the
            boundary.
        step : float, optional
            Finite-difference step for the marginal derivatives.

        Returns
        -------
        float
            Design value at which the marginal LQI margin is zero.
        """

        lower, upper = map(float, bounds)
        if lower >= upper:
            raise ValueError("bounds must be an increasing (lower, upper) pair")

        def margin(design: float) -> float:
            return self.acceptability_margin_at(
                safety_cost, failure_rate, design, step=step
            )

        lower_margin = margin(lower)
        if lower_margin == 0.0:
            return lower
        upper_margin = margin(upper)
        if upper_margin == 0.0:
            return upper
        if (lower_margin > 0.0) == (upper_margin > 0.0):
            raise ValueError(
                "marginal LQI margin does not change sign over bounds; "
                "no acceptability boundary in the given interval"
            )
        return float(brentq(margin, lower, upper))

    def evaluate(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return decision results with LQI/SWTP columns."""

        df = results.copy()

        if self.swtp is not None and self.consequence is not None:
            expected = self.consequence.expected_fatalities_given_failure
            df["expected_fatalities_given_failure"] = expected
            df["swtp_consequence"] = self.swtp.for_lives(expected)

        if self.target is not None:
            df["target_pf"] = self.target.pf
            df["target_beta"] = self.target.beta
            df["lqi_acceptable"] = df["pf"] <= self.target.pf
            # Screening slack: positive when pf is below the target.  Distinct
            # from the marginal-cost acceptability margin (acceptability_margin).
            df["screening_margin"] = self.target.pf - df["pf"]

        return df
