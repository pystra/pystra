"""Normalized code designs evaluated over self-weight and variable-load ratios."""

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from pandas import DataFrame

from ..assessment import ReliabilityEvaluator, ReliabilityResult, evaluate_reliability
from ..dependence.copula import Copula
from ..distributions import Constant, Distribution
from ..model import LimitState, StochasticModel
from ..options import FORMOptions
from ..reporting import reliability_row

__all__ = [
    "CodeFactors",
    "NominalValues",
    "NormalizedReliabilityModel",
    "CodeCalibration",
    "CodeDesignResult",
    "CodeCalibrationResult",
]


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def _ratios(values, name):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or not values.size
        or not np.all(np.isfinite(values))
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError(f"{name} must be a nonempty one-dimensional grid in [0, 1]")
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class CodeFactors:
    """Candidate normalized design factors (phi, gamma_g, gamma_p, gamma_q).

    All values must be finite and positive. No fitting objective or target
    reliability is implied by a factor set.
    """

    phi: float
    gamma_g: float
    gamma_p: float
    gamma_q: float

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(
                self, field.name, _positive(getattr(self, field.name), field.name)
            )


@dataclass(frozen=True)
class NominalValues:
    """Positive characteristic values in the normalized design equation."""

    resistance: float
    dead_load: float
    permanent_load: float
    live_load: float

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(
                self, field.name, _positive(getattr(self, field.name), field.name)
            )


@dataclass(frozen=True)
class NormalizedReliabilityModel:
    """Probability model and characteristic values, independent of code factors.

    R/G/P/Q distributions use the same normalization as ``nominal_values``;
    model errors may have nonunit means. Inputs are copied. Their names are
    user-controlled: the normalized equation binds quantities by their roles.
    ``copula`` defaults to independent marginals and, when supplied, follows
    the random-variable order of resistance_error, resistance, load_error,
    dead_load, permanent_load, live_load (constants omitted).
    """

    resistance: Union[Distribution, Constant]
    dead_load: Union[Distribution, Constant]
    permanent_load: Union[Distribution, Constant]
    live_load: Union[Distribution, Constant]
    resistance_error: Union[Distribution, Constant]
    load_error: Union[Distribution, Constant]
    nominal_values: NominalValues
    copula: Optional[Copula] = None

    def __post_init__(self):
        if not isinstance(self.nominal_values, NominalValues):
            raise TypeError("nominal_values must be NominalValues")
        names = []
        for name in self._roles:
            value = getattr(self, name)
            if not isinstance(value, (Distribution, Constant)):
                raise TypeError(f"{name} must be a Distribution or Constant")
            if isinstance(value, Constant) and not np.all(
                np.isfinite(np.asarray(value.value, dtype=float))
            ):
                raise ValueError(f"{name} must have a finite value")
            object.__setattr__(self, name, deepcopy(value))
            names.append(value.name)
        if len(set(names)) != len(names):
            raise ValueError(
                "Normalized reliability model variable names must be unique"
            )
        if self.copula is not None:
            object.__setattr__(self, "copula", deepcopy(self.copula))
        self.stochastic_model()  # validate dependence and at least one random input

    _roles = (
        "resistance_error",
        "resistance",
        "load_error",
        "dead_load",
        "permanent_load",
        "live_load",
    )

    def stochastic_model(self) -> StochasticModel:
        """Build an independent copy of this probability model."""
        model = StochasticModel()
        for name in self._roles:
            model.add_variable(deepcopy(getattr(self, name)))
        if not model.n_marg:
            raise ValueError("Reliability analysis needs at least one random variable")
        if self.copula is not None:
            model.set_copula(deepcopy(self.copula))
        return model


@dataclass(frozen=True)
class CodeDesignResult:
    """One normalized code design and its reliability result."""

    live_load_ratio: float
    dead_load_ratio: float
    design_value: float
    reliability: ReliabilityResult
    target_margin: Optional[float]


@dataclass(frozen=True)
class CodeCalibrationResult:
    """Snapshot of a factor-set study, with every grid point retained.

    Cases are ordered by dead-load ratio then live-load ratio. ``beta`` is a
    fresh array with shape (n_dead_ratios, n_live_ratios); failed cases are NaN.
    The probability-model snapshot is held privately and returned by copy.
    """

    live_load_ratios: Tuple[float, ...]
    dead_load_ratios: Tuple[float, ...]
    factors: CodeFactors
    target_beta: Optional[float]
    cases: Tuple[CodeDesignResult, ...]
    _model: NormalizedReliabilityModel

    @property
    def model(self) -> NormalizedReliabilityModel:
        """Copy of the probability model used for this result."""
        return deepcopy(self._model)

    @property
    def converged(self) -> bool:
        """Whether every grid-point analysis converged."""
        return all(case.reliability.converged for case in self.cases)

    @property
    def beta(self) -> np.ndarray:
        """Normal-equivalent indices; failed cases are explicitly NaN."""
        return np.array(
            [
                c.reliability.beta if c.reliability.converged else np.nan
                for c in self.cases
            ]
        ).reshape(len(self.dead_load_ratios), len(self.live_load_ratios))

    def to_frame(self) -> DataFrame:
        """Return an independent table including designs and failure status."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "live_load_ratio": c.live_load_ratio,
                    "dead_load_ratio": c.dead_load_ratio,
                    "design_value": c.design_value,
                    **reliability_row(c.reliability),
                    "target_margin": c.target_margin,
                }
                for c in self.cases
            ]
        )


class CodeCalibration:
    """Evaluate candidate factors on the normalized G/P/Q code-design grid.

    Parameters
    ----------
    live_load_ratios, dead_load_ratios : sequence of float
        Nonempty one-dimensional grids in [0, 1]. They are copied to tuples.

    Notes
    -----
    ``run`` returns a new result; no model registration or result cache is
    retained. Supply the same model with different CodeFactors to compare
    candidates. Automated factor fitting is deliberately a separate operation.
    """

    def __init__(
        self, *, live_load_ratios: Sequence[float], dead_load_ratios: Sequence[float]
    ) -> None:
        self._live_load_ratios = _ratios(live_load_ratios, "live_load_ratios")
        self._dead_load_ratios = _ratios(dead_load_ratios, "dead_load_ratios")

    @property
    def live_load_ratios(self) -> Tuple[float, ...]:
        """Immutable live-load grid coordinates, in evaluation order."""
        return self._live_load_ratios

    @property
    def dead_load_ratios(self) -> Tuple[float, ...]:
        """Immutable dead-load grid coordinates, in evaluation order."""
        return self._dead_load_ratios

    @staticmethod
    def design(
        *,
        live_load_ratio: float,
        dead_load_ratio: float,
        factors: CodeFactors,
        nominal_values: NominalValues,
    ) -> float:
        """Solve the normalized factored code equation for resistance scale z."""
        aq = _ratios([live_load_ratio], "live_load_ratio")[0]
        ag = _ratios([dead_load_ratio], "dead_load_ratio")[0]
        if not isinstance(factors, CodeFactors) or not isinstance(
            nominal_values, NominalValues
        ):
            raise TypeError("Expected CodeFactors and NominalValues")
        n, f = nominal_values, factors
        return (
            (1 - aq)
            * (ag * f.gamma_g * n.dead_load + (1 - ag) * f.gamma_p * n.permanent_load)
            + aq * f.gamma_q * n.live_load
        ) / (f.phi * n.resistance)

    def run(
        self,
        model: NormalizedReliabilityModel,
        factors: CodeFactors,
        *,
        options: object = None,
        evaluator: ReliabilityEvaluator | None = None,
        target_beta: Optional[float] = None,
    ) -> CodeCalibrationResult:
        """Evaluate candidate factors, retaining every failed grid point.

        Parameters
        ----------
        model : NormalizedReliabilityModel
            Probability inputs, nominal values and dependence, copied per run.
        factors : CodeFactors
            Candidate dimensionless factors in the normalized design equation.
        options : object, optional
            Evaluator settings; FORMOptions for the default method. DDM is not
            supported by the built-in normalized limit-state equation.
        evaluator : ReliabilityEvaluator, optional
            Method constructor or result callback; defaults to FORM. A design
            point is not required. See pystra.assessment.evaluate_reliability.
        target_beta : float, optional
            Finite normal-equivalent index for the same event/reference period.

        Returns
        -------
        CodeCalibrationResult
            Input snapshot and every design/result, ordered by dead-load ratio
            then live-load ratio. Failed table estimates and target margins are
            unavailable; original method records retain their diagnostics.

        Notes
        -----
        AnalysisError and nonconvergence are recorded. Invalid specifications
        and programming errors still raise. Beta and target margins are
        normal-equivalent, including for Student-t standard space.
        """
        if not isinstance(model, NormalizedReliabilityModel) or not isinstance(
            factors, CodeFactors
        ):
            raise TypeError("Expected NormalizedReliabilityModel and CodeFactors")
        if isinstance(options, FORMOptions) and options.differentiation == "ddm":
            raise ValueError(
                "The normalized study currently requires finite-difference derivatives"
            )
        if target_beta is not None:
            target_beta = float(target_beta)
            if not np.isfinite(target_beta):
                raise ValueError("target_beta must be finite")
        snapshot = deepcopy(model)
        cases = []
        role_names = {role: getattr(snapshot, role).name for role in snapshot._roles}
        for ag in self.dead_load_ratios:
            for aq in self.live_load_ratios:
                z = self.design(
                    live_load_ratio=aq,
                    dead_load_ratio=ag,
                    factors=factors,
                    nominal_values=snapshot.nominal_values,
                )

                def limit_state(**values):
                    wR, R, wS, G, P, Q = (
                        values[role_names[role]] for role in snapshot._roles
                    )
                    return z * wR * R - wS * (
                        (1 - aq) * (ag * G + (1 - ag) * P) + aq * Q
                    )

                result = evaluate_reliability(
                    snapshot.stochastic_model(),
                    LimitState(limit_state),
                    options=options,
                    evaluator=evaluator,
                )
                margin = (
                    result.beta - target_beta
                    if result.converged and target_beta is not None
                    else None
                )
                cases.append(CodeDesignResult(aq, ag, z, result, margin))
        return CodeCalibrationResult(
            self.live_load_ratios,
            self.dead_load_ratios,
            factors,
            target_beta,
            tuple(cases),
            snapshot,
        )
