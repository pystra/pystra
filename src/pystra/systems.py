"""System reliability limit-state composition.

This module provides a small topology layer for structural system
reliability. It composes component functions into a scalar limit-state function
for simulation. The separate SystemFORM class estimates series and parallel
probabilities from component tangent planes.

The sign convention is the standard PySTRA convention: positive values are
safe, and negative values indicate failure.  A series system fails when any
child fails, so its equivalent limit-state value is the minimum child value.
A parallel system fails when all children fail, so its equivalent
limit-state value is the maximum child value.
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .model import LimitState

__all__ = [
    "Component",
    "System",
    "SeriesSystem",
    "ParallelSystem",
    "KOfNSystem",
    "CutSetSystem",
    "TieSetSystem",
    "ditlevsen_bounds",
]


class Component:
    """A named component limit state within a structural system.

    Parameters
    ----------
    name : str or LimitState or callable
        Component name.  For convenience, if *limit_state* is omitted this
        argument is treated as the limit-state object and the name is inferred.
    limit_state : LimitState or callable, optional
        Component limit-state function.  Callables are wrapped in
        :class:`~pystra.model.LimitState`.

    Notes
    -----
    Component functions may use only a subset of the stochastic model
    variables.  Extra keyword arguments are filtered unless the function
    accepts ``**kwargs``.
    """

    def __init__(
        self,
        name: str | LimitState | Callable[..., Any] | None,
        limit_state: LimitState | Callable[..., Any] | None = None,
    ) -> None:
        if limit_state is None:
            limit_state = name
            name = None

        self.limit_state = _as_limit_state(limit_state)
        self.name = _component_name(name, self.limit_state)
        self._signature = _call_signature(self.limit_state.expression)

    def evaluate(self, **kwargs: ArrayLike) -> np.ndarray:
        """Evaluate the component limit state for one or more samples."""

        values = self.limit_state.expression(**self._filter_kwargs(kwargs))
        if isinstance(values, tuple):
            values = values[0]
        return np.asarray(values, dtype=float)

    def as_limit_state(self) -> LimitState:
        """Return this component as a PySTRA :class:`LimitState`."""

        # Keep tuple returns intact for direct differentiation. User-supplied
        # gradients must follow the complete stochastic model variable order.
        return LimitState(
            lambda **kwargs: self.limit_state.expression(**self._filter_kwargs(kwargs))
        )

    def get_limit_state(self) -> LimitState:
        """Return this component as a PySTRA :class:`LimitState`.

        This legacy-style alias mirrors the existing PySTRA getter naming.
        """

        return self.as_limit_state()

    def iter_components(self) -> Iterator[Component]:
        """Yield leaf components in this subtree."""

        yield self

    @property
    def components(self) -> tuple[Component, ...]:
        """Tuple containing this component."""

        return (self,)

    def failure_mask(self, **kwargs: ArrayLike) -> np.bool_ | np.ndarray:
        """Return a boolean mask where the component is failed."""

        return self.evaluate(**kwargs) < 0

    def _filter_kwargs(self, kwargs):
        names, required, accepts_kwargs = self._signature
        if accepts_kwargs:
            return kwargs

        missing = [name for name in required if name not in kwargs]
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(
                f"Component '{self.name}' is missing required input(s): "
                f"{missing_text}"
            )
        return {name: kwargs[name] for name in names if name in kwargs}


class System:
    """Base class for composed structural system limit states."""

    _operator_name = "system"

    def __init__(
        self,
        children: Iterable[Component | System | LimitState | Callable[..., Any]],
        name: str | None = None,
    ) -> None:
        children = tuple(_as_node(child, index) for index, child in enumerate(children))
        if not children:
            raise ValueError("A system must contain at least one child")

        self.children = children
        self.name = name or self._operator_name

    @property
    def components(self) -> tuple[Component, ...]:
        """Flat tuple of all leaf components in the system."""

        return tuple(_unique_components(self.iter_components()))

    def iter_components(self) -> Iterator[Component]:
        """Yield all leaf components in the system."""

        for child in self.children:
            yield from child.iter_components()

    def evaluate(self, **kwargs: ArrayLike) -> float | np.ndarray:
        """Evaluate the equivalent scalar system limit state."""

        values = [child.evaluate(**kwargs) for child in self.children]
        return self._combine(values)

    def as_limit_state(self) -> LimitState:
        """Return the composed system as a PySTRA :class:`LimitState`."""

        return LimitState(lambda **kwargs: self.evaluate(**kwargs))

    def get_limit_state(self) -> LimitState:
        """Return the composed system as a PySTRA :class:`LimitState`.

        This legacy-style alias mirrors the existing PySTRA getter naming.
        """

        return self.as_limit_state()

    def component_values(self, **kwargs: ArrayLike) -> dict[str, np.ndarray]:
        """Evaluate all leaf component limit states.

        Returns
        -------
        dict
            Mapping component names to their evaluated limit-state values.

        Raises
        ------
        ValueError
            If duplicate component names would make the mapping ambiguous.
        """

        values = {}
        for component in self.components:
            if component.name in values:
                raise ValueError(
                    f"Duplicate component name '{component.name}' in system"
                )
            values[component.name] = component.evaluate(**kwargs)
        return values

    def component_failure_masks(
        self, **kwargs: ArrayLike
    ) -> dict[str, np.bool_ | np.ndarray]:
        """Return failure masks for each leaf component."""

        return {
            name: values < 0 for name, values in self.component_values(**kwargs).items()
        }

    def failure_mask(self, **kwargs: ArrayLike) -> np.bool_ | np.ndarray:
        """Return a boolean mask where the system is failed."""

        return self.evaluate(**kwargs) < 0

    def _combine(self, values):
        raise NotImplementedError


class SeriesSystem(System):
    """A system that fails when any child component or subsystem fails.

    The equivalent limit-state function is the minimum child limit-state
    value.
    """

    _operator_name = "series"

    def _combine(self, values):
        arrays = _broadcast_values(values)
        return np.min(arrays, axis=0)


class ParallelSystem(System):
    """A system that fails when all child components or subsystems fail.

    The equivalent limit-state function is the maximum child limit-state
    value.
    """

    _operator_name = "parallel"

    def _combine(self, values):
        arrays = _broadcast_values(values)
        return np.max(arrays, axis=0)


class KOfNSystem(System):
    """A system that fails when at least ``k`` child events fail.

    Parameters
    ----------
    children : iterable
        Child components or subsystems.
    k : int
        Number of failed children required for system failure.
    name : str, optional
        System name.

    Notes
    -----
    This event-counting representation is intended for simulation,
    enumeration, and topology construction.  Its equivalent limit-state value
    preserves the failure sign but is discontinuous, so it is generally not a
    smooth FORM/SORM limit state.
    """

    _operator_name = "k_of_n"

    def __init__(
        self,
        children: Iterable[Component | System | LimitState | Callable[..., Any]],
        k: int,
        name: str | None = None,
    ) -> None:
        super().__init__(children, name=name)
        self.k = int(k)
        if self.k < 1 or self.k > len(self.children):
            raise ValueError("k must satisfy 1 <= k <= number of children")

    def _combine(self, values):
        arrays = _broadcast_values(values)
        failed_count = np.count_nonzero(arrays < 0.0, axis=0)
        return self.k - failed_count - 0.5


class CutSetSystem(System):
    """A system described by cut sets.

    Each cut set is a group of component failures that causes system failure.
    The system fails when any cut set has fully failed.

    Parameters
    ----------
    cut_sets : iterable of iterables
        Cut sets.  Entries may be components, subsystems, callables, or string
        names when *components* is supplied.
    components : mapping or iterable, optional
        Component catalogue used to resolve string names in *cut_sets*.
    name : str, optional
        System name.
    """

    _operator_name = "cut_set"

    def __init__(
        self,
        cut_sets: Iterable[
            Iterable[str | Component | System | LimitState | Callable[..., Any]]
        ],
        components: (
            dict[str, Component | System | LimitState | Callable[..., Any]]
            | Iterable[Component | System | LimitState | Callable[..., Any]]
            | None
        ) = None,
        name: str | None = None,
    ) -> None:
        resolved = _resolve_component_sets(cut_sets, components)
        if not resolved:
            raise ValueError("At least one cut set is required")

        self.cut_sets = resolved
        children = [ParallelSystem(cut_set) for cut_set in resolved]
        super().__init__(children, name=name)

    def _combine(self, values):
        arrays = _broadcast_values(values)
        return np.min(arrays, axis=0)


class TieSetSystem(System):
    """A system described by tie sets.

    Each tie set is a group of components that keeps the system safe when all
    components in that set are safe.  The system fails only when all tie sets
    have failed.

    Parameters
    ----------
    tie_sets : iterable of iterables
        Tie sets.  Entries may be components, subsystems, callables, or string
        names when *components* is supplied.
    components : mapping or iterable, optional
        Component catalogue used to resolve string names in *tie_sets*.
    name : str, optional
        System name.
    """

    _operator_name = "tie_set"

    def __init__(
        self,
        tie_sets: Iterable[
            Iterable[str | Component | System | LimitState | Callable[..., Any]]
        ],
        components: (
            dict[str, Component | System | LimitState | Callable[..., Any]]
            | Iterable[Component | System | LimitState | Callable[..., Any]]
            | None
        ) = None,
        name: str | None = None,
    ) -> None:
        resolved = _resolve_component_sets(tie_sets, components)
        if not resolved:
            raise ValueError("At least one tie set is required")

        self.tie_sets = resolved
        children = [SeriesSystem(tie_set) for tie_set in resolved]
        super().__init__(children, name=name)

    def _combine(self, values):
        arrays = _broadcast_values(values)
        return np.max(arrays, axis=0)


def ditlevsen_bounds(
    probabilities: ArrayLike,
    intersections: ArrayLike | dict[tuple[int, int], float],
    ordering: Sequence[int] | np.ndarray | None = None,
    optimize_order: bool = False,
) -> tuple[float, float]:
    r"""Return Ditlevsen bounds for the probability of a union of events.

    Parameters
    ----------
    probabilities : array_like
        Event probabilities :math:`P(E_i)`.
    intersections : array_like or mapping
        Pairwise intersection probabilities :math:`P(E_i \cap E_j)`.  This may
        be an ``n x n`` array, or a mapping keyed by ``(i, j)`` pairs.
    ordering : sequence of int, optional
        Event ordering used by Ditlevsen's sequential bounds.
    optimize_order : bool, optional
        If ``True``, all event orderings are checked and the tightest lower and
        upper bounds are returned.  This is factorial in the number of events
        and is limited to eight events.

    Returns
    -------
    tuple of float
        Lower and upper bounds for :math:`P(\cup_i E_i)`.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional sequence")
    if len(probabilities) == 0:
        raise ValueError("At least one event probability is required")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probabilities must be between 0 and 1")

    pairwise = _intersection_matrix(intersections, len(probabilities))
    for i in range(len(probabilities)):
        for j in range(i):
            lo = max(0.0, probabilities[i] + probabilities[j] - 1.0)
            hi = min(probabilities[i], probabilities[j])
            if not lo - 1e-14 <= pairwise[i, j] <= hi + 1e-14:
                raise ValueError("intersections violate marginal probability bounds")

    if optimize_order:
        if ordering is not None:
            raise ValueError("ordering cannot be supplied with optimize_order=True")
        if len(probabilities) > 8:
            raise ValueError("optimize_order=True is limited to eight events")
        bounds = [
            _ditlevsen_bounds_for_order(probabilities, pairwise, candidate)
            for candidate in itertools.permutations(range(len(probabilities)))
        ]
        return max(lower for lower, _ in bounds), min(upper for _, upper in bounds)

    if ordering is None:
        ordering = tuple(range(len(probabilities)))
    return _ditlevsen_bounds_for_order(probabilities, pairwise, ordering)


def _as_node(obj, index):
    if isinstance(obj, (Component, System)):
        return obj
    return Component(f"component_{index}", obj)


def _as_limit_state(obj):
    if isinstance(obj, LimitState):
        return obj
    if callable(obj):
        return LimitState(obj)
    raise TypeError("limit_state must be a LimitState or callable")


def _component_name(name, limit_state):
    if name is not None:
        return str(name)

    expression = limit_state.expression
    inferred = getattr(expression, "__name__", None)
    if inferred and inferred != "<lambda>":
        return inferred
    return "component"


def _call_signature(expression: Callable):
    signature = inspect.signature(expression)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    names = tuple(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    required = tuple(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and parameter.default is inspect.Parameter.empty
    )
    return names, required, accepts_kwargs


def _broadcast_values(values):
    if not values:
        raise ValueError("At least one value is required")
    arrays = [np.asarray(value, dtype=float) for value in values]
    return np.stack(np.broadcast_arrays(*arrays), axis=0)


def _unique_components(components):
    seen = set()
    unique = []
    for component in components:
        identifier = id(component)
        if identifier not in seen:
            seen.add(identifier)
            unique.append(component)
    return unique


def _resolve_component_sets(component_sets, components):
    catalogue = _component_catalogue(components)
    resolved_sets = []
    for component_set in component_sets:
        resolved = []
        for component_index, component in enumerate(component_set):
            if isinstance(component, str):
                if component not in catalogue:
                    raise KeyError(f"Unknown component name '{component}'")
                resolved.append(catalogue[component])
            else:
                resolved.append(_as_node(component, component_index))
        if not resolved:
            raise ValueError("Component sets cannot be empty")
        resolved_sets.append(tuple(resolved))
    return tuple(resolved_sets)


def _component_catalogue(components):
    if components is None:
        return {}
    if isinstance(components, dict):
        return {
            str(name): _as_node(component, index)
            for index, (name, component) in enumerate(components.items())
        }

    catalogue = {}
    for index, component in enumerate(components):
        node = _as_node(component, index)
        if node.name in catalogue:
            raise ValueError(f"Duplicate component name '{node.name}' in catalogue")
        catalogue[node.name] = node
    return catalogue


def _intersection_matrix(intersections, n_events):
    if isinstance(intersections, dict):
        matrix = np.zeros((n_events, n_events), dtype=float)
        for i in range(n_events):
            for j in range(i):
                if (i, j) not in intersections and (j, i) not in intersections:
                    raise ValueError("Every pairwise intersection must be supplied")
        for key, value in intersections.items():
            i, j = key
            if not (0 <= i < n_events and 0 <= j < n_events):
                raise ValueError("Intersection index out of range")
            if (j, i) in intersections and intersections[j, i] != value:
                raise ValueError("intersections must be symmetric")
            matrix[i, j] = matrix[j, i] = float(value)
    else:
        matrix = np.asarray(intersections, dtype=float)

    if matrix.shape != (n_events, n_events):
        raise ValueError("intersections must be an n x n matrix or pair mapping")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("intersections must be between 0 and 1")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("intersections must be symmetric")
    return matrix


def _ditlevsen_bounds_for_order(probabilities, intersections, ordering):
    order = tuple(ordering)
    n_events = len(probabilities)
    if sorted(order) != list(range(n_events)):
        raise ValueError("ordering must be a permutation of event indices")

    lower = probabilities[order[0]]
    upper = np.sum(probabilities)
    for position, event in enumerate(order[1:], start=1):
        previous = order[:position]
        previous_intersections = intersections[event, previous]
        lower += max(probabilities[event] - np.sum(previous_intersections), 0.0)
        upper -= np.max(previous_intersections)

    if lower > upper + 1e-12:
        raise ValueError("Inconsistent probabilities produce reversed bounds")
    return _clip_probability(lower), _clip_probability(upper)


def _clip_probability(value):
    return float(np.clip(value, 0.0, 1.0))
