"""Target design solving and specialist design-point factor derivation.

The coefficient and matrix methods preserve the Caprani and Khan (Structural
Safety, 2023) reference cases. They assume a separable resistance-scale design
rule, including multiplicative resistance/load model errors. They are not a
general decomposition of arbitrary nonlinear limit states.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Optional, Tuple, Mapping, Union
from pandas import DataFrame

import numpy as np
from scipy.optimize import fsolve, root_scalar

from ..analysis import AnalysisOptions
from ..distributions import Constant, Distribution
from ..form import Form
from ..loadcomb import LoadCombination
from ..model import LimitState
from ..results import FormResult

__all__ = [
    "FactorCalibrationProblem",
    "CalibratedDesign",
    "TargetDesigns",
    "FactorSet",
    "GoverningFactor",
    "DesignValues",
    "DesignVerification",
    "analyze_case",
    "solve_designs",
    "derive_factors",
    "select_factors",
    "design_with_factors",
    "verify_designs",
]


def _scalar(value, name):
    values = np.asarray(value, dtype=float)
    if values.size != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite scalar")
    return float(values.item())


def _run_form(cases, case_name, overrides=None, options=None):
    if not isinstance(cases, LoadCombination):
        raise TypeError("cases must be LoadCombination")
    if cases.limit_state is None:
        raise ValueError("A limit_state is required for reliability evaluation")
    if options is not None and not isinstance(options, AnalysisOptions):
        raise TypeError("options must be AnalysisOptions")
    solver = Form(
        cases.stochastic_model(case_name, overrides=overrides),
        LimitState(cases.limit_state),
        deepcopy(options),
    )
    result = solver.run()
    return solver, result


def analyze_case(
    cases: LoadCombination,
    case_name: Optional[str] = None,
    *,
    overrides: Optional[Mapping[str, Union[Distribution, Constant]]] = None,
    options: Optional[AnalysisOptions] = None,
) -> FormResult:
    """Evaluate one explicit load case and return a FORM snapshot."""
    return _run_form(cases, case_name, overrides, options)[1]


class FactorCalibrationProblem:
    """Explicit inputs to the separable resistance-scale factor methods.

    Parameters
    ----------
    cases : LoadCombination
        Cases with explicit roles and leading-action metadata.
    nominal_values : mapping
        Positive nominal values for exactly the random-variable role names.
    design_parameter : str
        Name of a common Constant multiplying the resistance term.

    Inputs are snapshotted. Numerical operations use named coordinates and
    arrays; no DataFrame carries state between stages. The default design
    rule is z = load_effect / resistance_effect, evaluated at unit z with the
    opposite side set to zero. Every computed design is checked in the full
    limit state to reject incompatible decompositions.
    """

    def __init__(
        self,
        cases: LoadCombination,
        *,
        nominal_values: Mapping[str, float],
        design_parameter: str,
    ) -> None:
        if not isinstance(cases, LoadCombination) or cases.roles is None:
            raise ValueError(
                "Factor calibration requires cases with explicit variable roles"
            )
        if (
            not cases.roles.resistance
            or not cases.roles.variable
            or not cases.leading_actions
        ):
            raise ValueError(
                "Resistance, variable actions and leading actions must be specified"
            )
        if cases.limit_state is None:
            raise ValueError("A limit_state is required")
        if design_parameter not in cases.constants:
            raise ValueError("design_parameter must name a common Constant")
        if set(nominal_values) != set(cases.roles.names):
            raise ValueError(
                "Nominal values must match exactly the random-variable roles"
            )
        self._cases = deepcopy(cases)
        self._nominal_values = {
            name: _scalar(nominal_values[name], name) for name in cases.roles.names
        }
        if any(v <= 0 for v in self._nominal_values.values()):
            raise ValueError("Nominal values must be positive")
        self.design_parameter = design_parameter

    @property
    def cases(self) -> LoadCombination:
        """Copy of the snapshotted cases."""
        return deepcopy(self._cases)

    @property
    def nominal_values(self) -> Mapping[str, float]:
        """Read-only nominal values in role order."""
        return MappingProxyType(self._nominal_values)

    def _effect(self, case_name, values, *, design_value=1.0):
        model = self._cases.stochastic_model(case_name)
        unknown = set(values) - set(model.names)
        if unknown:
            raise ValueError(f"Unknown design-point names: {sorted(unknown)}")
        arguments = {name: 0.0 for name in model.get_variables()}
        arguments.update(model.constants)
        arguments[self.design_parameter] = design_value
        arguments.update(values)
        return _scalar(self._cases.limit_state(**arguments), "limit-state value")

    def design_from_point(self, case_name: str, values: Mapping[str, float]) -> float:
        """Solve the separable design rule and verify its full residual."""
        roles = self._cases.roles
        if set(values) != set(roles.names):
            raise ValueError("Design point must contain exactly the role names")
        resistance = self._effect(case_name, {n: values[n] for n in roles.resistance})
        load = self._effect(
            case_name, {n: values[n] for n in roles.other + roles.variable}
        )
        if resistance == 0:
            raise ValueError("Zero resistance effect in design rule")
        z = abs(load / resistance)
        residual = self._effect(case_name, values, design_value=z)
        if not np.isfinite(z) or not np.isclose(
            residual, 0, atol=1e-7 * max(1, abs(load), abs(z * resistance)), rtol=0
        ):
            raise ValueError(
                "Limit state does not satisfy the separable resistance-scale design rule"
            )
        return z


@dataclass(frozen=True)
class CalibratedDesign:
    """Target solve outcome; unsuccessful designs cannot enter factor derivation."""

    case_name: str
    design_value: float
    target_beta: float
    reliability: Optional[FormResult]
    residual: Optional[float]
    converged: bool
    evaluations: int
    message: str


@dataclass(frozen=True)
class TargetDesigns:
    """Target designs and the input problem used to obtain them."""

    designs: Tuple[CalibratedDesign, ...]
    _problem: FactorCalibrationProblem

    @property
    def problem(self) -> FactorCalibrationProblem:
        return deepcopy(self._problem)

    @property
    def converged(self) -> bool:
        return all(d.converged for d in self.designs)

    def to_frame(self) -> DataFrame:
        """Fresh table of physical design points plus the named design parameter."""
        import pandas as pd

        if not self.converged:
            raise ValueError("Not all target designs converged")
        roles = self._problem._cases.roles
        rows = []
        for d in self.designs:
            point = dict(zip(d.reliability.variable_names, d.reliability.design_point))
            rows.append([point[n] for n in roles.names] + [d.design_value])
        return pd.DataFrame(
            rows,
            index=[d.case_name for d in self.designs],
            columns=roles.names + (self._problem.design_parameter,),
        )


class _InnerFailure(Exception):
    pass


def solve_designs(
    problem: FactorCalibrationProblem,
    *,
    target_beta: float,
    method: str = "root",
    initial_value: Optional[float] = None,
    tolerance: float = 0.0001,
    max_evaluations: int = 100,
    bracket: Optional[Tuple[float, float]] = None,
    options: Optional[AnalysisOptions] = None,
) -> TargetDesigns:
    """Solve each case to a target and return status/residuals for every case.

    ``root`` uses fsolve with full diagnostics, or bracketed Brent solving when
    a bracket is supplied. ``alpha`` preserves the normal-space projection
    method. Both require converged inner FORM results and a final beta residual
    within ``tolerance``. ``max_evaluations`` limits FORM runs per case,
    including the final verification. No unsuccessful solve is silently accepted.
    """
    if not isinstance(problem, FactorCalibrationProblem):
        raise TypeError("Expected FactorCalibrationProblem")
    if method not in ("root", "alpha"):
        raise ValueError("method must be 'root' or 'alpha'")
    target = _scalar(target_beta, "target_beta")
    tolerance = _scalar(tolerance, "tolerance")
    if (
        tolerance <= 0
        or isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 1
    ):
        raise ValueError("Positive tolerance and integer evaluation budget required")
    if bracket is not None:
        if method != "root" or len(bracket) != 2:
            raise ValueError("bracket requires two endpoints and the root method")
        bracket = tuple(_scalar(v, "bracket") for v in bracket)
        if bracket[0] >= bracket[1]:
            raise ValueError("bracket endpoints must be increasing")
    snapshot = deepcopy(problem)
    start = _scalar(
        (
            snapshot._cases.constants[snapshot.design_parameter].get_value()
            if initial_value is None
            else initial_value
        ),
        "initial_value",
    )
    outcomes = []
    for case_name in snapshot._cases.case_names:
        evaluations = 0
        result = None
        solver = None
        value = start

        def evaluate(candidate):
            nonlocal evaluations, result, solver, value
            if evaluations >= max_evaluations:
                raise _InnerFailure("FORM evaluation budget exhausted")
            value = _scalar(candidate, "design value")
            evaluations += 1
            solver, result = _run_form(
                snapshot._cases,
                case_name,
                {snapshot.design_parameter: Constant(snapshot.design_parameter, value)},
                options,
            )
            if not result.converged:
                raise _InnerFailure("Inner FORM did not converge")
            return result.beta - target

        successful = False
        message = "Target solve did not converge"
        try:
            if method == "root":
                if bracket is None:
                    root, diagnostics, status, message = fsolve(
                        evaluate,
                        start,
                        xtol=tolerance,
                        maxfev=max_evaluations,
                        full_output=True,
                    )
                    candidate = float(root[0])
                    successful = status == 1
                else:
                    lower, upper = (evaluate(v) for v in bracket)
                    if lower * upper > 0:
                        raise _InnerFailure("Target is not bracketed")
                    solution = root_scalar(
                        evaluate,
                        bracket=bracket,
                        xtol=tolerance,
                        maxiter=max_evaluations,
                    )
                    candidate, successful, message = (
                        solution.root,
                        solution.converged,
                        str(solution.flag),
                    )
                residual = evaluate(candidate)
            else:
                residual = evaluate(start)
                if result.standard_space != "normal":
                    raise ValueError("Alpha projection requires normal standard space")
                for _ in range(max_evaluations):
                    if abs(residual) <= tolerance:
                        successful = True
                        break
                    u = np.asarray(result.alpha) * target
                    x = solver.transform.u_to_x(
                        u, solver.model.get_marginal_distributions()
                    )
                    point = dict(zip(result.variable_names, np.asarray(x).ravel()))
                    candidate = snapshot.design_from_point(case_name, point)
                    residual = evaluate(candidate)
                successful = abs(residual) <= tolerance
                message = (
                    "Converged"
                    if successful
                    else "Alpha projection iteration limit reached"
                )
        except _InnerFailure as error:
            successful = False
            message = str(error)
        residual = (
            result.beta - target if result is not None and result.converged else None
        )
        if successful and residual is not None and abs(residual) > tolerance:
            message = f"Target residual exceeds tolerance: {residual:g}"
        successful = bool(
            successful and residual is not None and abs(residual) <= tolerance
        )
        outcomes.append(
            CalibratedDesign(
                case_name,
                value,
                target,
                result,
                residual,
                successful,
                evaluations,
                "Converged" if successful else message,
            )
        )
    return TargetDesigns(tuple(outcomes), snapshot)


@dataclass(frozen=True)
class GoverningFactor:
    """Provenance for an extremal factor selection (ties retained)."""

    kind: str
    variable: str
    case_names: Tuple[str, ...]


@dataclass(frozen=True)
class FactorSet:
    """Immutable per-case factors with named axes and selection provenance.

    Numeric matrices are tuples of rows; ``to_frame`` produces independent
    reporting tables. Resistance columns use ``resistance_names``; load and
    combination columns use ``load_names``. Leading actions are explicit.
    """

    case_names: Tuple[str, ...]
    resistance_names: Tuple[str, ...]
    load_names: Tuple[str, ...]
    resistance: Tuple[Tuple[float, ...], ...]
    loads: Tuple[Tuple[float, ...], ...]
    combinations: Tuple[Tuple[float, ...], ...]
    leading_actions: Tuple[Tuple[str, ...], ...]
    governing: Tuple[GoverningFactor, ...] = ()

    def __post_init__(self):
        for field in ("case_names", "resistance_names", "load_names", "governing"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if len(set(self.case_names)) != len(self.case_names) or not self.case_names:
            raise ValueError("Factor cases must be nonempty and unique")
        if len(set(self.resistance_names + self.load_names)) != len(
            self.resistance_names + self.load_names
        ):
            raise ValueError("Factor variable names must be unique and disjoint")
        for field, names in (
            ("resistance", self.resistance_names),
            ("loads", self.load_names),
            ("combinations", self.load_names),
        ):
            array = np.asarray(getattr(self, field), dtype=float)
            if array.shape != (len(self.case_names), len(names)) or not np.all(
                np.isfinite(array)
            ):
                raise ValueError(
                    "Factor matrix shape or values do not match named axes"
                )
            object.__setattr__(
                self, field, tuple(tuple(float(x) for x in row) for row in array)
            )
        object.__setattr__(
            self, "leading_actions", tuple(tuple(row) for row in self.leading_actions)
        )
        if len(self.leading_actions) != len(self.case_names) or any(
            not set(row) <= set(self.load_names) for row in self.leading_actions
        ):
            raise ValueError("Leading actions do not match factor axes")

    def to_frame(self, kind: str) -> DataFrame:
        """Return a fresh resistance, load, or combination factor table."""
        import pandas as pd

        if kind not in ("resistance", "loads", "combinations"):
            raise ValueError("Unknown factor kind")
        names = self.resistance_names if kind == "resistance" else self.load_names
        return pd.DataFrame(getattr(self, kind), index=self.case_names, columns=names)


def derive_factors(solutions: TargetDesigns, *, method: str = "matrix") -> FactorSet:
    """Derive coefficient or matrix factors from fully converged target designs.

    The specialist formulation requires one leading case per variable action.
    Case and variable order may differ. All assembly uses leading-action names.
    Singular systems and unsupported case topologies fail explicitly.
    """
    if method not in ("matrix", "coeff"):
        raise ValueError("method must be 'matrix' or 'coeff'")
    if not solutions.converged:
        raise ValueError("Cannot derive factors from failed target designs")
    problem = solutions._problem
    cases, roles = problem._cases, problem._cases.roles
    leading = cases.leading_actions
    if any(len(row) != 1 for row in leading.values()) or sorted(
        row[0] for row in leading.values()
    ) != sorted(roles.variable):
        raise ValueError(
            "Factor derivation needs exactly one leading case per variable action"
        )
    names = roles.names
    points = np.array(
        [
            [
                dict(zip(d.reliability.variable_names, d.reliability.design_point))[n]
                for n in names
            ]
            for d in solutions.designs
        ]
    )
    if any(d.reliability.standard_space != "normal" for d in solutions.designs):
        raise ValueError(
            "Design-point factor derivation requires normal standard space"
        )
    nominals = np.array([problem.nominal_values[n] for n in names])
    normalized = points / nominals
    for action in roles.variable:
        lead_row = next(
            i for i, n in enumerate(cases.case_names) if action in leading[n]
        )
        normalized[:, names.index(action)] = normalized[lead_row, names.index(action)]
    n_resistance = len(roles.resistance)
    loads = normalized[:, n_resistance:]
    combinations = points[:, n_resistance:] / loads / nominals[n_resistance:]
    if method == "matrix":
        size = len(roles.variable)
        matrix, rhs = np.zeros((size, size)), np.zeros(size)
        for i, design in enumerate(solutions.designs):
            point = dict(zip(names, points[i]))
            lead = leading[design.case_name]
            included = roles.resistance + roles.other + lead
            rhs[i] = problem._effect(
                design.case_name,
                {n: point[n] for n in included},
                design_value=design.design_value,
            )
        # Each action coefficient is evaluated in its own leading case,
        # preserving the reference method's column-wise effect convention.
        # Names determine both columns and the omitted leading entries.
        for j, action in enumerate(roles.variable):
            lead_row = next(
                i for i, n in enumerate(cases.case_names) if action in leading[n]
            )
            lead_case = cases.case_names[lead_row]
            factored = dict(zip(names, normalized[lead_row] * nominals))
            other = {n: factored[n] for n in roles.other}
            coefficient = -(
                problem._effect(lead_case, {**other, action: factored[action]})
                - problem._effect(lead_case, other)
            )
            for i, case_name in enumerate(cases.case_names):
                if action not in leading[case_name]:
                    matrix[i, j] = coefficient
        try:
            psi = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError("Singular combination-factor system") from error
        combinations = np.ones_like(loads)
        for i, case_name in enumerate(cases.case_names):
            for j, action in enumerate(roles.variable):
                combinations[i, len(roles.other) + j] = (
                    1 if action in leading[case_name] else psi[j]
                )
    return FactorSet(
        cases.case_names,
        roles.resistance,
        roles.other + roles.variable,
        normalized[:, :n_resistance],
        loads,
        combinations,
        tuple(leading[n] for n in cases.case_names),
    )


def select_factors(
    factors: FactorSet,
    *,
    resistance: str = "per_case",
    loads: str = "per_case",
    combinations: str = "per_case",
) -> FactorSet:
    """Select explicit extrema or retain per-case factors, recording governing cases.

    Leading-action combination factors remain one. Companion maxima exclude
    leading cases by name. Selection does not certify achieved reliability.
    """
    policies = (
        ("resistance", resistance, "minimum"),
        ("loads", loads, "maximum"),
        ("combinations", combinations, "maximum"),
    )
    selected, governing = {}, []
    for kind, policy, allowed in policies:
        if policy not in ("per_case", allowed):
            raise ValueError(f"{kind} policy must be 'per_case' or '{allowed}'")
        values = np.array(getattr(factors, kind), copy=True)
        names = factors.resistance_names if kind == "resistance" else factors.load_names
        if policy != "per_case":
            for j, name in enumerate(names):
                eligible = [
                    i
                    for i, lead in enumerate(factors.leading_actions)
                    if kind != "combinations" or name not in lead
                ]
                if eligible:
                    candidates = values[eligible, j]
                    extreme = float(
                        np.min(candidates)
                        if policy == "minimum"
                        else np.max(candidates)
                    )
                    sources = tuple(
                        factors.case_names[i]
                        for i in eligible
                        if values[i, j] == extreme
                    )
                    governing.append(GoverningFactor(kind, name, sources))
                    values[eligible, j] = extreme
                if kind == "combinations":
                    for i, lead in enumerate(factors.leading_actions):
                        if name in lead:
                            values[i, j] = 1
        selected[kind] = values
    return replace(factors, **selected, governing=tuple(governing))


@dataclass(frozen=True)
class DesignValues:
    """Factored design parameters in named case order."""

    case_names: Tuple[str, ...]
    values: Tuple[float, ...]

    @property
    def governing_cases(self) -> Tuple[str, ...]:
        """Cases attaining the largest required resistance scale."""
        maximum = max(self.values)
        return tuple(n for n, v in zip(self.case_names, self.values) if v == maximum)


def design_with_factors(
    problem: FactorCalibrationProblem, factors: FactorSet
) -> DesignValues:
    """Apply a selected factor set to the nominal design rule, by case/name."""
    roles = problem._cases.roles
    if (
        set(factors.case_names) != set(problem._cases.case_names)
        or set(factors.resistance_names) != set(roles.resistance)
        or set(factors.load_names) != set(roles.other + roles.variable)
    ):
        raise ValueError("Factor axes do not match the design problem")
    for i, name in enumerate(factors.case_names):
        if set(factors.leading_actions[i]) != set(problem._cases.leading_actions[name]):
            raise ValueError("Factor leading actions do not match the design problem")
    values = []
    for case in problem._cases.case_names:
        i = factors.case_names.index(case)
        point = {
            n: problem.nominal_values[n] * factors.resistance[i][j]
            for j, n in enumerate(factors.resistance_names)
        }
        point.update(
            {
                n: problem.nominal_values[n]
                * factors.loads[i][j]
                * factors.combinations[i][j]
                for j, n in enumerate(factors.load_names)
            }
        )
        values.append(problem.design_from_point(case, point))
    return DesignValues(problem._cases.case_names, tuple(values))


@dataclass(frozen=True)
class DesignVerification:
    """Achieved reliability and optional target margin for one supplied design."""

    case_name: str
    design_value: float
    reliability: FormResult
    target_margin: Optional[float]


def verify_designs(
    problem: FactorCalibrationProblem,
    design_values: Union[float, Mapping[str, float], DesignValues],
    *,
    target_beta: Optional[float] = None,
    options: Optional[AnalysisOptions] = None,
) -> Tuple[DesignVerification, ...]:
    """Check a common design scale or an explicitly named set of case designs.

    To check the governing common design, supply ``max(designs.values)``
    explicitly. Every case is returned, including nonconverged analyses.
    """
    names = problem._cases.case_names
    if isinstance(design_values, DesignValues):
        design_values = dict(zip(design_values.case_names, design_values.values))
    if hasattr(design_values, "keys"):
        if set(design_values) != set(names):
            raise ValueError("Design values must cover exactly the cases")
        values = {n: _scalar(design_values[n], n) for n in names}
    else:
        values = {n: _scalar(design_values, "design value") for n in names}
    target = None if target_beta is None else _scalar(target_beta, "target_beta")
    results = []
    for name, value in values.items():
        result = analyze_case(
            problem._cases,
            name,
            overrides={
                problem.design_parameter: Constant(problem.design_parameter, value)
            },
            options=options,
        )
        margin = (
            result.beta - target if result.converged and target is not None else None
        )
        results.append(DesignVerification(name, value, result, margin))
    return tuple(results)
