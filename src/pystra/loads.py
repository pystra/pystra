"""Load processes and inspectable probabilistic load cases.

Ferry-Borges-Castanheta rectangular-wave processes, explicit load cases and
leading-action case generators.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Tuple, Optional, Callable, Union, Sequence

import numpy as np

from .distributions import Constant, Distribution, Maximum, MaxParent
from .model import StochasticModel
from .errors import ModelError

__all__ = ["FBCProcess", "VariableRoles", "LoadCombination"]


class FBCProcess:
    """Ferry-Borges-Castanheta rectangular-wave load process.

    The Ferry-Borges-Castanheta (FBC) model represents a load process as a
    sequence of independent rectangular pulses of equal duration.  Within
    each basic interval the process is constant; between intervals a new
    value is drawn from the interval parent distribution.

    This class is intentionally a small distribution factory.  It does not
    run a reliability analysis and does not create load combinations by
    itself.  It only returns ordinary Pystra distribution objects that can be
    added to a :class:`~pystra.model.StochasticModel` or used in
    :class:`~pystra.loads.LoadCombination` cases.

    Load combinations are a separate modelling choice.  For example,
    :meth:`pystra.loads.LoadCombination.turkstra` uses FBC process objects
    to generate leading-action cases according to Turkstra's rule.

    Parameters
    ----------
    name : str
        Name of the load process.  This must match the name of ``parent`` and
        the corresponding argument in the limit-state function.
    parent : Distribution
        Distribution of the process value in one basic interval.
    basic_interval : float
        Duration of one rectangular-wave interval, in the same time units used
        for ``duration`` arguments passed to :meth:`maximum`.

    Notes
    -----
    For a duration :math:`T` and basic interval :math:`\\tau`, the maximum is
    represented by ``Maximum(parent, N=T/tau)``.  Durations shorter than one
    basic interval are treated as one interval.

    Examples
    --------
    >>> import pystra as ra
    >>> Q = ra.FBCProcess(
    ...     "Q", parent=ra.Gumbel("Q", 0.89, 0.2), basic_interval=1/52
    ... )
    >>> point_in_time = Q.point_in_time()
    >>> maximum = Q.maximum(duration=50)
    """

    def __init__(self, name, parent, basic_interval):
        if not isinstance(parent, Distribution):
            raise ModelError("FBCProcess parent must be a Pystra Distribution")
        if basic_interval <= 0:
            raise ModelError("FBCProcess basic_interval must be positive")
        if parent.get_name() != name:
            raise ModelError("FBCProcess name must match parent distribution name")

        self.name = name
        self.parent = parent
        self.basic_interval = basic_interval

    @classmethod
    def from_maximum(cls, name, maximum, maximum_duration, basic_interval):
        """Create a process from a known maximum distribution.

        This is useful when a code or model supplies, for example, an annual
        maximum distribution but the FBC process requires the distribution of
        one basic interval.  The interval parent is inferred using
        :class:`~pystra.distributions.parent.MaxParent`.

        Parameters
        ----------
        name : str
            Process and random-variable name.
        maximum : Distribution
            Distribution of the maximum over ``maximum_duration``.
        maximum_duration : float
            Duration represented by ``maximum``.
        basic_interval : float
            Basic interval of the rectangular-wave process.

        Returns
        -------
        FBCProcess
            Process whose parent distribution is inferred from ``maximum``.
        """
        if maximum_duration <= 0:
            raise ModelError("FBCProcess maximum_duration must be positive")
        if basic_interval <= 0:
            raise ModelError("FBCProcess basic_interval must be positive")
        if not isinstance(maximum, Distribution):
            raise ModelError("FBCProcess maximum must be a Pystra Distribution")
        if maximum.get_name() != name:
            raise ModelError("FBCProcess name must match maximum distribution name")

        n = max(1.0, maximum_duration / basic_interval)
        parent = MaxParent(name, maximum, N=n)
        return cls(name, parent, basic_interval)

    def interval_count(self, duration=None, n=None):
        """Return the number of basic intervals in a duration.

        Either ``duration`` or ``n`` may be supplied, but not both.  Durations
        shorter than one basic interval are treated as one interval, which is
        the parent distribution.

        Parameters
        ----------
        duration : float, optional
            Duration over which the process is observed.
        n : float, optional
            Direct number of basic intervals.

        Returns
        -------
        float
            Number of basic intervals used in the maximum distribution.
        """
        if (duration is None) == (n is None):
            raise ModelError("Specify exactly one of duration or n")
        if n is not None:
            if n < 1.0:
                raise ModelError("FBCProcess n must be >= 1.0")
            return n
        if duration <= 0:
            raise ModelError("FBCProcess duration must be positive")
        return max(1.0, duration / self.basic_interval)

    def point_in_time(self):
        """Return the interval parent distribution.

        Returns
        -------
        Distribution
            Distribution of the process value in one basic interval.
        """
        return self.parent

    def maximum(self, duration=None, n=None):
        """Return the distribution of the process maximum.

        Parameters
        ----------
        duration : float, optional
            Duration over which the maximum is taken.
        n : float, optional
            Direct number of basic intervals over which the maximum is taken.

        Returns
        -------
        Maximum
            Distribution of the maximum over the requested duration or number
            of intervals.
        """
        n_intervals = self.interval_count(duration=duration, n=n)
        return Maximum(self.name, self.parent, N=n_intervals)


_Variables = Union[
    Mapping[str, Union[Distribution, Constant]], Sequence[Union[Distribution, Constant]]
]


def _variables(values, *, processes=False):
    if values is None:
        return {}
    entries = (
        values.items() if isinstance(values, Mapping) else ((v.name, v) for v in values)
    )
    result = {}
    allowed = (
        (Distribution, Constant, FBCProcess) if processes else (Distribution, Constant)
    )
    for name, variable in entries:
        if not isinstance(variable, allowed):
            raise TypeError(
                "Expected a Distribution or Constant"
                + (" or FBCProcess" if processes else "")
            )
        if name != variable.name or name in result:
            raise ValueError(
                "Variable names must be unique and agree with mapping keys"
            )
        result[name] = deepcopy(variable)
    return result


@dataclass(frozen=True)
class VariableRoles:
    """Names of resistance, static/other, and variable-action quantities.

    Model errors belong to the resistance or other group according to the
    side of the limit state they multiply. Ordering aligns numerical arrays;
    leading actions are specified separately for each case.
    """

    resistance: Tuple[str, ...] = ()
    other: Tuple[str, ...] = ()
    variable: Tuple[str, ...] = ()

    def __post_init__(self):
        for field in ("resistance", "other", "variable"):
            values = getattr(self, field)
            if isinstance(values, str):
                raise TypeError("Variable roles require sequences of names")
            object.__setattr__(self, field, tuple(values))
        if any(not isinstance(n, str) or not n for n in self.names):
            raise ValueError("Variable names must be nonempty strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Variable roles must be disjoint and unique")

    @property
    def names(self) -> Tuple[str, ...]:
        """All role names in resistance/other/variable order."""
        return self.resistance + self.other + self.variable


class LoadCombination:
    """Named probabilistic cases, with optional factor-calibration metadata.

    Parameters
    ----------
    cases : mapping
        Case names mapped to named distributions/constants or sequences thereof.
    limit_state : callable, optional
        Limit-state function. No analysis is run by this object.
    constants : mapping or sequence, optional
        Common constants; case variables may not duplicate these names.
    roles : VariableRoles, optional
        Explicit roles, required by specialist factor calibration.
    leading_actions : mapping, optional
        Case names mapped to tuples of leading variable-action names.
    correlation : pandas.DataFrame, optional
        Physical Pearson correlations labelled by random-variable names.

    Notes
    -----
    Input objects are copied. Accessors and model construction return copies,
    so candidate overrides cannot mutate the stored case definitions.
    """

    def __init__(
        self,
        *,
        cases: Mapping[str, _Variables],
        limit_state: Optional[Callable] = None,
        constants: Optional[_Variables] = None,
        roles: Optional[VariableRoles] = None,
        leading_actions: Optional[Mapping[str, Sequence[str]]] = None,
        correlation=None,
    ) -> None:
        if not isinstance(cases, Mapping) or not cases:
            raise ValueError("At least one named case is required")
        if any(not isinstance(n, str) or not n for n in cases):
            raise ValueError("Case names must be nonempty strings")
        if limit_state is not None and not callable(limit_state):
            raise TypeError("limit_state must be callable")
        self._limit_state = limit_state
        self._cases = {name: _variables(values) for name, values in cases.items()}
        self._constants = _variables(constants)
        if any(not isinstance(v, Constant) for v in self._constants.values()):
            raise TypeError("Common constants must be Constant objects")
        for case in self._cases.values():
            if set(case) & set(self._constants):
                raise ValueError("Case variables duplicate common constants")
        if roles is not None and not isinstance(roles, VariableRoles):
            raise TypeError("roles must be VariableRoles")
        self._roles = roles
        if roles is not None:
            for case in self._cases.values():
                random_names = {
                    n for n, v in case.items() if isinstance(v, Distribution)
                }
                if random_names != set(roles.names):
                    raise ValueError(
                        "Roles must cover exactly the random variables in every case"
                    )
        self._leading_actions = {}
        if leading_actions is not None:
            if roles is None or set(leading_actions) != set(cases):
                raise ValueError(
                    "Leading actions require roles and an entry for every case"
                )
            for name, values in leading_actions.items():
                if isinstance(values, str):
                    raise TypeError("Leading actions must be sequences of names")
                values = tuple(values)
                if (
                    not values
                    or len(set(values)) != len(values)
                    or not set(values) <= set(roles.variable)
                ):
                    raise ValueError(
                        "Leading actions must identify unique variable actions"
                    )
                self._leading_actions[name] = values
        self._correlation = deepcopy(correlation)
        # Validate all case-dependent correlation subsets before a study starts.
        for name in self.case_names:
            self.stochastic_model(name)

    @property
    def limit_state(self) -> Optional[Callable]:
        """Failure-event callable supplied with the case specification."""
        return self._limit_state

    @property
    def roles(self) -> Optional[VariableRoles]:
        """Validated, immutable resistance/other/action metadata."""
        return self._roles

    @property
    def case_names(self) -> Tuple[str, ...]:
        """Stable case order."""
        return tuple(self._cases)

    @property
    def cases(self) -> dict:
        """Independent copies of all cases."""
        return deepcopy(self._cases)

    @property
    def constants(self) -> dict:
        """Independent copies of common constants."""
        return deepcopy(self._constants)

    @property
    def leading_actions(self) -> Mapping[str, Tuple[str, ...]]:
        """Read-only case-to-leading-action metadata."""
        return MappingProxyType(self._leading_actions)

    def case(self, case_name: Optional[str] = None) -> dict:
        """Return a copy of one case; default to the first case."""
        name = self.case_names[0] if case_name is None else case_name
        if name not in self._cases:
            raise ValueError(f"Unknown case: {name}")
        return deepcopy(self._cases[name])

    def stochastic_model(
        self, case_name: Optional[str] = None, *, overrides: Optional[_Variables] = None
    ) -> StochasticModel:
        """Build an isolated model; reject unknown or misnamed overrides."""
        variables = {**deepcopy(self._constants), **self.case(case_name)}
        replacements = _variables(overrides)
        unknown = set(replacements) - set(variables)
        if unknown:
            raise ValueError(f"Unknown overrides: {sorted(unknown)}")
        variables.update(replacements)
        model = StochasticModel()
        for variable in variables.values():
            model.add_variable(variable)
        if self._correlation is not None:
            names = tuple(model.get_variables())
            corr = self._correlation
            if (
                not hasattr(corr, "reindex")
                or not corr.index.is_unique
                or not corr.columns.is_unique
            ):
                raise ValueError("correlation must be a uniquely labelled DataFrame")
            if not set(names) <= set(corr.index) or not set(names) <= set(corr.columns):
                raise ValueError("Missing correlation labels")
            values = corr.reindex(index=names, columns=names).to_numpy(dtype=float)
            if (
                not np.all(np.isfinite(values))
                or not np.allclose(values, values.T)
                or not np.allclose(np.diag(values), 1)
            ):
                raise ValueError(
                    "Correlation must be finite, symmetric, with unit diagonal"
                )
            if len(names) and np.min(np.linalg.eigvalsh(values)) <= 0:
                raise ValueError("Correlation must be positive definite")
            model.set_correlation(values.copy())
        return model

    @classmethod
    def from_actions(
        cls,
        *,
        maxima: Mapping[str, Distribution],
        companions: Mapping[str, Distribution],
        resistance: _Variables,
        other: Optional[_Variables] = None,
        constants: Optional[_Variables] = None,
        leading_actions: Optional[Mapping[str, Sequence[str]]] = None,
        limit_state: Optional[Callable] = None,
        correlation=None,
    ) -> "LoadCombination":
        """Generate cases from explicitly supplied maximum/companion marginals.

        Both mappings use the action names as keys. With no leading metadata,
        generate one case per action, named ``<action>_max``. No reference-period
        conversion is inferred; use :meth:`turkstra` for FBC processes.
        """
        maxima, companions = _variables(maxima), _variables(companions)
        if not maxima or set(maxima) != set(companions):
            raise ValueError(
                "Maximum and companion actions must have identical nonempty names"
            )
        if any(
            not isinstance(v, Distribution)
            for v in (*maxima.values(), *companions.values())
        ):
            raise TypeError("Actions must be distributions")
        resistance, other = _variables(resistance), _variables(other)
        roles = VariableRoles(tuple(resistance), tuple(other), tuple(maxima))
        leading = (
            leading_actions
            if leading_actions is not None
            else {f"{n}_max": (n,) for n in maxima}
        )
        cases = {
            case: {
                **resistance,
                **other,
                **{
                    n: value if n in lead else companions[n]
                    for n, value in maxima.items()
                },
            }
            for case, lead in leading.items()
        }
        return cls(
            cases=cases,
            limit_state=limit_state,
            constants=constants,
            roles=roles,
            leading_actions=leading,
            correlation=correlation,
        )

    @classmethod
    def turkstra(
        cls,
        variable: Mapping[str, FBCProcess],
        reference_period: float,
        *,
        limit_state: Optional[Callable] = None,
        resistance: Optional[_Variables] = None,
        permanent: Optional[_Variables] = None,
        other: Optional[_Variables] = None,
        constants: Optional[_Variables] = None,
        correlation=None,
        companion_duration: Union[str, float] = "leading_interval",
    ) -> "LoadCombination":
        """Generate FBC leading/companion cases using Turkstra's rule.

        The leading marginal is the maximum over ``reference_period``. A
        companion uses the leading process's basic interval by default, or
        ``'point_in_time'`` or an explicitly supplied duration. Units must agree.
        """
        if not np.isfinite(reference_period) or reference_period <= 0:
            raise ValueError("reference_period must be finite and positive")
        variable = _variables(variable, processes=True)
        if not variable or any(
            not isinstance(v, FBCProcess) for v in variable.values()
        ):
            raise TypeError("Variable actions must be FBCProcess objects")
        resistance, permanent, other = (
            _variables(resistance),
            _variables(permanent),
            _variables(other),
        )
        if set(permanent) & set(other):
            raise ValueError("Permanent and other variables overlap")
        common = {**resistance, **permanent, **other}
        roles = VariableRoles(
            tuple(resistance), tuple(permanent) + tuple(other), tuple(variable)
        )
        cases = {}
        for lead_name, lead in variable.items():
            case = dict(common)
            for name, process in variable.items():
                if name == lead_name:
                    value = process.maximum(duration=reference_period)
                elif companion_duration == "leading_interval":
                    value = process.maximum(duration=lead.basic_interval)
                elif companion_duration == "point_in_time":
                    value = process.point_in_time()
                else:
                    value = process.maximum(duration=companion_duration)
                case[name] = value
            cases[f"{lead_name}_leading"] = case
        return cls(
            cases=cases,
            limit_state=limit_state,
            constants=constants,
            roles=roles,
            leading_actions={f"{n}_leading": (n,) for n in variable},
            correlation=correlation,
        )
