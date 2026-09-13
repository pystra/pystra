"""Isolated case evaluation for assessment, calibration and design studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import numpy as np
from pandas import DataFrame

from .distributions import Constant, Distribution
from .errors import AnalysisError
from .loads import LoadCombination
from .model import LimitState, StochasticModel
from .options import FORMOptions
from .reliability.form import FORM
from .reporting import _probabilities, reliability_row
from .results import (
    FORMResult,
    SORMResult,
    SimulationResult,
    SystemFORMResult,
    SensitivityResult,
)

__all__ = [
    "ReliabilityResult",
    "ReliabilityEvaluator",
    "ReliabilityEstimate",
    "AssessmentCase",
    "CaseResult",
    "AssessmentResult",
    "evaluate_reliability",
    "analyze_case",
    "assess_cases",
]


class ReliabilityResult(Protocol):
    """Minimum result contract for a reliability study.

    Probability and beta refer to the same failure event and reference period;
    beta is normal-equivalent. A failed analysis may retain an estimate, but
    study calculations use it only when ``converged`` is true. A design point
    is a separate capability and is not implied by this protocol.
    """

    failure_probability: float | None
    beta: float | None
    method: str
    status: str
    message: str

    @property
    def converged(self) -> bool: ...


class ReliabilityEvaluator(Protocol):
    """Callback receiving isolated inputs and returning a result or analysis.

    The call is ``evaluator(model, limit_state, options=options)``. A method
    constructor such as ``CrudeMonteCarlo`` may be supplied directly; its
    ``run()`` is called once. An external-solver adapter may instead return a
    ``ReliabilityEstimate`` or another record meeting ``ReliabilityResult``.
    The callback owns its random state and any external resources.
    """

    def __call__(
        self, model: StochasticModel, limit_state: LimitState, *, options: Any = None
    ) -> Any: ...


@dataclass(frozen=True, kw_only=True)
class ReliabilityEstimate:
    """Small result record for an external or analytic reliability evaluator.

    Parameters
    ----------
    method : str
        Evaluator name recorded in study tables.
    failure_probability, beta : float or None, optional
        Dimensionless probability and normal-equivalent index. A successful
        result requires at least one; the other is derived. Supply both to
        preserve a finite tail index when probability underflows to zero.
    status : str, default "completed"
        ``converged`` or ``completed`` indicates success. ``not_converged``
        and ``precision_not_met`` retain failed or under-resolved estimates.
    message : str, default ""
        Solver termination or precision diagnostics.
    """

    method: str
    failure_probability: float | None = None
    beta: float | None = None
    status: str = "completed"
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a nonempty name")
        if self.status not in (
            "converged",
            "completed",
            "not_converged",
            "precision_not_met",
        ):
            raise ValueError("Unknown reliability status")
        if (
            self.converged
            or self.failure_probability is not None
            or self.beta is not None
        ):
            pf, beta = _probabilities(self.failure_probability, self.beta)
            object.__setattr__(self, "failure_probability", pf)
            object.__setattr__(self, "beta", beta)

    @property
    def converged(self) -> bool:
        """Whether the evaluator met its termination criteria."""
        return self.status in ("converged", "completed")


def _snapshot_result(result: Any) -> Any:
    """Retain immutable built-in records; isolate external mutable results."""
    if type(result) in (
        ReliabilityEstimate,
        FORMResult,
        SORMResult,
        SimulationResult,
        SystemFORMResult,
        SensitivityResult,
    ):
        return result
    return deepcopy(result)


def _evaluation_method(evaluator: Any, error: AnalysisError) -> str:
    """Identify the known solver or callback behind a resultless failure."""
    if evaluator is None:
        return "FORM"
    while isinstance(evaluator, partial):
        evaluator = evaluator.func
    owner = getattr(evaluator, "__self__", None)
    if owner is not None:
        evaluator = owner
    name = getattr(evaluator, "__name__", type(evaluator).__name__)
    if name == "<lambda>":
        return f"external evaluator ({type(error).__name__})"
    return name


def _evaluate(
    model: StochasticModel,
    limit_state: LimitState,
    *,
    options: Any = None,
    evaluator: ReliabilityEvaluator | None = None,
) -> tuple[Any, ReliabilityResult]:
    """Run once and retain the analysis only for design-point projection."""
    if not isinstance(model, StochasticModel) or not isinstance(
        limit_state, LimitState
    ):
        raise TypeError("Expected StochasticModel and LimitState")
    if evaluator is None:
        if options is not None and not isinstance(options, FORMOptions):
            raise TypeError("options must be FORMOptions for the default evaluator")
    elif not callable(evaluator):
        raise TypeError("evaluator must be callable")
    analysis = None
    try:
        inputs = deepcopy(model), deepcopy(limit_state)
        if evaluator is None:
            candidate = FORM(*inputs, options=deepcopy(options), on_failure="return")
        else:
            candidate = evaluator(*inputs, options=deepcopy(options))
        if callable(getattr(candidate, "run", None)):
            analysis = candidate
            result = candidate.run()
        else:
            result = candidate
    except AnalysisError as error:
        result = error.result
        if result is None:
            result = ReliabilityEstimate(
                method=_evaluation_method(
                    analysis if analysis is not None else evaluator, error
                ),
                status="not_converged",
                message=str(error),
            )
        elif getattr(result, "converged", False):
            raise ValueError("AnalysisError must carry a failed result") from error
    required = (
        "failure_probability",
        "beta",
        "converged",
        "status",
        "message",
        "method",
    )
    if not all(hasattr(result, name) for name in required):
        raise TypeError(
            "evaluator must return an analysis or a ReliabilityResult record"
        )
    reliability_row(result)  # Validate before any study consumes the estimate.
    if result.converged and (result.beta is None or result.failure_probability is None):
        raise TypeError(
            "A successful ReliabilityResult must provide both beta and failure_probability; use ReliabilityEstimate to derive either"
        )
    # Built-in results already own their data; copy external mutable records too.
    return analysis, _snapshot_result(result)


def evaluate_reliability(
    model: StochasticModel,
    limit_state: LimitState,
    *,
    options: Any = None,
    evaluator: ReliabilityEvaluator | None = None,
) -> ReliabilityResult:
    """Evaluate isolated inputs, defaulting to FORM, and return a snapshot.

    Parameters
    ----------
    model, limit_state : StochasticModel, LimitState
        Probability model and failure event. Inputs are copied before evaluation.
    options : object, optional
        Settings passed to the evaluator; FORMOptions for the default FORM.
    evaluator : ReliabilityEvaluator, optional
        Method constructor or callback. Its result follows ReliabilityResult.

    Returns
    -------
    ReliabilityResult
        Success or failure record, including diagnostics. AnalysisError is
        recorded; invalid specifications and programming errors still raise.
    """
    return _evaluate(model, limit_state, options=options, evaluator=evaluator)[1]


def analyze_case(
    cases: LoadCombination,
    case_name: str | None = None,
    *,
    overrides: Mapping[str, Distribution | Constant] | None = None,
    options: Any = None,
    evaluator: ReliabilityEvaluator | None = None,
) -> ReliabilityResult:
    """Evaluate a named load case with isolated overrides and return its record.

    Parameters
    ----------
    cases : LoadCombination
        Explicit probabilistic cases with a limit-state callable.
    case_name : str, optional
        Case name; defaults to the first case.
    overrides : mapping, optional
        Named replacement distributions/constants; unknown names are rejected.
    options, evaluator : optional
        Settings and callback as in :func:`evaluate_reliability`; default FORM.

    Returns
    -------
    ReliabilityResult
        Snapshot retaining success or failure status and diagnostics.
    """
    if not isinstance(cases, LoadCombination):
        raise TypeError("cases must be LoadCombination")
    if cases.limit_state is None:
        raise ValueError("A limit_state is required for reliability evaluation")
    return evaluate_reliability(
        cases.stochastic_model(case_name, overrides=overrides),
        LimitState(cases.limit_state),
        options=options,
        evaluator=evaluator,
    )


class AssessmentCase:
    """Named physical scenario and its engineering assumptions.

    Parameters
    ----------
    name : str
        Unique label within an assessment.
    model, limit_state : StochasticModel, LimitState
        Inputs for this scenario, copied on construction and access.
    metadata : mapping, optional
        Engineering provenance, for example units, reference period, dependence,
        target source and model-error assumptions. Copied without interpretation;
        assessment never assumes temporal independence or converts periods.
    """

    def __init__(
        self,
        name: str,
        model: StochasticModel,
        limit_state: LimitState,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Case name must be a nonempty string")
        if not isinstance(model, StochasticModel) or not isinstance(
            limit_state, LimitState
        ):
            raise TypeError("Expected StochasticModel and LimitState")
        self._name = name
        self._model = deepcopy(model)
        self._limit_state = deepcopy(limit_state)
        self._metadata = deepcopy(dict(metadata or {}))

    @property
    def name(self) -> str:
        """Scenario label."""
        return self._name

    @property
    def model(self) -> StochasticModel:
        """Independent copy of the scenario probability model."""
        return deepcopy(self._model)

    @property
    def limit_state(self) -> LimitState:
        """Independent copy of the failure-event specification."""
        return deepcopy(self._limit_state)

    @property
    def metadata(self) -> dict[str, Any]:
        """Copy of the engineering assumptions and provenance."""
        return deepcopy(self._metadata)


@dataclass(frozen=True)
class CaseResult:
    """Scenario, reliability snapshot and normal-equivalent target margin."""

    case: AssessmentCase
    reliability: ReliabilityResult
    target_margin: float | None


@dataclass(frozen=True)
class AssessmentResult:
    """Ordered case records, including every unsuccessful analysis."""

    cases: tuple[CaseResult, ...]
    target_beta: float | None

    @property
    def converged(self) -> bool:
        """Whether every case met its method's termination criteria."""
        return all(case.reliability.converged for case in self.cases)

    def to_frame(self) -> DataFrame:
        """Return a fresh table of estimates, status, margins and assumptions."""
        return DataFrame(
            [
                {
                    "case_name": case.case.name,
                    **reliability_row(case.reliability),
                    "target_margin": case.target_margin,
                    "metadata": case.case.metadata,
                }
                for case in self.cases
            ]
        )


def assess_cases(
    cases: Sequence[AssessmentCase],
    *,
    target_beta: float | None = None,
    options: Any = None,
    evaluator: ReliabilityEvaluator | None = None,
) -> AssessmentResult:
    """Compare named scenarios without imposing a load-combination schema.

    Parameters
    ----------
    cases : sequence of AssessmentCase
        Nonempty scenarios with unique names, evaluated in supplied order.
    target_beta : float, optional
        Finite normal-equivalent target for the same event/reference period.
    options, evaluator : optional
        Settings and callback as in :func:`evaluate_reliability`; default FORM.

    Returns
    -------
    AssessmentResult
        Input snapshots, method records and beta-minus-target margins. Failed
        analyses remain in the result and have no target margin.
    """
    cases = tuple(cases)
    if not cases or not all(isinstance(case, AssessmentCase) for case in cases):
        raise ValueError("Provide a nonempty sequence of AssessmentCase objects")
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("Assessment case names must be unique")
    if target_beta is not None:
        target_beta = float(target_beta)
        if not np.isfinite(target_beta):
            raise ValueError("target_beta must be finite")
    records = []
    for case in deepcopy(cases):
        result = evaluate_reliability(
            case.model, case.limit_state, options=options, evaluator=evaluator
        )
        margin = (
            result.beta - target_beta
            if result.converged and target_beta is not None
            else None
        )
        records.append(CaseResult(case, result, margin))
    return AssessmentResult(tuple(records), target_beta)
