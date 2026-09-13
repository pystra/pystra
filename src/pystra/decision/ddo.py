"""Design decision orchestration and established decision imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from .criteria import (
    DDOCriterion,
    LQI,
    _lqi_consequence,
    finite_difference_derivative,
    jcss_lqi_acceptability,
    jcss_lqi_acceptability_margin,
    jcss_lqi_is_acceptable,
)
from .objectives import (
    CostBenefitModel,
    DDOObjective,
    annualized_safety_cost,
    jcss_systematic_reconstruction_objective,
    present_value_factor,
)
from .plotting import (
    _coerce_reference_designs,
    _default_decision_plot_quantities,
    plot_summary,
)
from .risk import (
    FatalityConsequence,
    RiskResult,
    ScenarioRiskModel,
    jcss_lqi_risk_cost,
    jcss_lqi_risk_cost_from_result,
)
from .studies import DesignStudy, RiskStudy, _coerce_analysis_result
from .swtp import (
    SWTP,
    SWTPIndexRecord,
    SWTPRecord,
    SWTP_ALIASES,
    SWTP_COUNTRY_SOURCE,
    SWTP_COUNTRY_VALUES,
    SWTP_GDP_PPP_INDEX_2024,
    SWTP_INDEX_INDICATOR,
    SWTP_INDEX_SOURCE,
    SWTP_TARGET_SOURCE,
    _index,
    _index_table,
    _normalise_country_code,
    _record,
    _require_explicit_indexed,
    _swtp_value,
    get_swtp,
    get_swtp_index_record,
    get_swtp_record,
    index_swtp_record,
    swtp_table,
)
from .targets import (
    RackwitzTargetModel,
    TargetReliability,
    TargetReliabilityCalibration,
    _TARGET_TABLE,
    _VARIABILITY_FACTORS,
    _as_scalar_or_array,
    _beta_from_failure_probability,
    derive_lqi_target,
    lognormal_ratio_failure_probability,
    lqi_k1,
    lqi_target_reliability,
    rackwitz_table,
)


@dataclass
class DDO:
    """Evaluate a decision context with an objective and acceptability criterion.

    Construct directly from the three pieces::

        ddo = DDO(
            study=study,
            objective=CostBenefitModel(...),
            criterion=LQI.from_country(...),
        )

    Construction is keyword-only by design: ``study``, ``objective`` and
    ``criterion`` are easy to transpose positionally, which would silently
    misassign them.  :meth:`run` evaluates every alternative and caches the
    table; :meth:`optimize` returns the best feasible alternative and
    :meth:`economic_optimum` the unconstrained economic best.
    """

    study: DesignStudy | RiskStudy
    criterion: DDOCriterion
    objective: DDOObjective | None = None
    _results: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __init__(
        self,
        *,
        study: DesignStudy | RiskStudy,
        criterion: DDOCriterion,
        objective: DDOObjective | None = None,
    ):
        self.study = study
        self.criterion = criterion
        self.objective = objective
        self._results = None

    def _evaluate(self) -> pd.DataFrame:
        df = self.study.evaluate()
        if self.objective is not None:
            successful = self._successful(df)
            if successful.any():
                evaluated = self.objective.evaluate(
                    df.loc[successful].copy(), design=self.study.variable
                )
                df = pd.concat([evaluated, df.loc[~successful]]).reindex(df.index)
            else:
                df[self.objective.objective_column] = float("nan")
        return self.criterion.evaluate(df)

    @property
    def results(self) -> pd.DataFrame | None:
        """Independent table from the last completed run, or None."""
        return None if self._results is None else self._results.copy(deep=True)

    def run(self) -> pd.DataFrame:
        """Evaluate every alternative; a failed rerun clears the previous table."""
        self._results = None
        self._results = self._evaluate()
        return self.results

    def _results_or_run(self) -> pd.DataFrame:
        return self.results if self._results is not None else self.run()

    @staticmethod
    def _successful(results: pd.DataFrame) -> pd.Series:
        """Only successful reliability evaluations can support a decision."""
        if "converged" in results:
            return results["converged"].fillna(False).astype(bool)
        return pd.Series(True, index=results.index)

    def _objective_column(self) -> str:
        if self.objective is not None:
            return self.objective.objective_column
        return "objective"

    def economic_optimum(self) -> pd.Series:
        """Return the alternative with the largest objective, ignoring feasibility."""

        df = self._results_or_run()
        objective_column = self._objective_column()
        if objective_column not in df:
            raise ValueError("Objective results are not available")
        candidates = df.loc[self._successful(df) & df[objective_column].notna()]
        if candidates.empty:
            raise ValueError("No successful alternatives have an objective value")
        return candidates.loc[candidates[objective_column].idxmax()]

    def feasible_results(self) -> pd.DataFrame:
        """Return evaluated alternatives satisfying the criterion."""

        df = self._results_or_run()
        return df.loc[self._successful(df) & self.criterion.feasible(df).fillna(False)]

    def optimize(self) -> pd.Series:
        """Return the feasible alternative with the largest objective value."""

        feasible = self.feasible_results()
        if feasible.empty:
            raise ValueError("No feasible alternatives satisfy the criterion")
        objective_column = self._objective_column()
        if objective_column not in feasible:
            raise ValueError("Objective results are not available")
        return feasible.loc[feasible[objective_column].idxmax()]

    def summary(self) -> pd.DataFrame:
        """Return the key decision points as a small labeled table.

        One row for the unconstrained economic optimum and, when the criterion
        admits one, a second for the best feasible alternative.  Columns are the
        design variable, ``pf``, ``beta`` (when present), the objective, and the
        criterion's feasibility flag.  This is the table a designer reads off a
        study, without rebuilding it row by row.
        """

        df = self._results_or_run()
        objective_column = self._objective_column()
        if objective_column not in df:
            raise ValueError("DDO.summary requires an objective")

        points = [("economic optimum", self.economic_optimum())]
        feasible = self.feasible_results()
        if not feasible.empty:
            points.append(
                ("best feasible", feasible.loc[feasible[objective_column].idxmax()])
            )

        feasibility_column = getattr(self.criterion, "feasibility_column", None)
        candidate_columns = [
            self.study.variable,
            "pf",
            "beta",
            objective_column,
            feasibility_column,
        ]
        columns = [
            column
            for column in candidate_columns
            if column is not None and column in df
        ]
        rows = [
            {"point": label, **{c: row[c] for c in columns}} for label, row in points
        ]
        return pd.DataFrame(rows, columns=["point"] + columns)

    def plot(
        self,
        design: str | None = None,
        quantities: Sequence[str] | None = None,
        **kwargs,
    ):
        """Plot results from the most recent run."""

        df = self._results_or_run()
        design_column = self.study.variable if design is None else design
        return plot_summary(df, design=design_column, quantities=quantities, **kwargs)


__all__ = [
    # Societal value of life
    "SWTP",
    "FatalityConsequence",
    # Acceptability criterion and its result type
    "LQI",
    "TargetReliability",
    # Objective
    "CostBenefitModel",
    # Studies
    "DesignStudy",
    "RiskStudy",
    "ScenarioRiskModel",
    "RiskResult",
    # Orchestration
    "DDO",
    "DDOObjective",
    "DDOCriterion",
    # Code-calibration target model
    "RackwitzTargetModel",
]
