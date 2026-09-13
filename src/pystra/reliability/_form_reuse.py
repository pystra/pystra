"""Validate reuse of a completed FORM calculation without numerical work."""

from collections.abc import Mapping

import numpy as np

from ..dependence.copula import Copula
from ..distributions.distribution import Distribution, rv_frozen
from ..errors import AnalysisError, ModelError

__all__ = []


class _FORMReuse:
    """Keep explicit FORM assignment distinct from an internally computed FORM."""

    @property
    def form(self):
        """The last FORM analysis; assigning None requests a fresh FORM each run."""
        return self._form

    @form.setter
    def form(self, value):
        from .form import FORM

        if value is not None and not isinstance(value, FORM):
            raise TypeError("form must be a FORM analysis")
        self._form = value
        self._supplied_form = value
        self._results_valid = False


def _state(value):
    """Snapshot parameter data, excluding SciPy's unrelated random state.

    Unknown extension state cannot establish safe reuse; the caller then
    recomputes FORM. No serialization of user objects is required.
    """
    if value is None or isinstance(value, (str, bool, int, float, np.number)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays require recomputation")
        return value.dtype.str, value.shape, value.tobytes()
    if isinstance(value, Mapping):
        return tuple((key, _state(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_state(item) for item in value)
    if isinstance(value, rv_frozen):
        return type(value.dist), value.dist.name, _state(value.args), _state(value.kwds)
    if isinstance(value, (Distribution, Copula)) and type(value).__module__.startswith(
        "pystra."
    ):
        return type(value), _state(vars(value))
    raise TypeError("Unknown parameter state requires recomputation")


def _problem_state(model):
    """Record the model data used by FORM, excluding evaluation counters."""
    try:
        return _state(
            (
                model.get_marginal_distributions(),
                tuple(model.get_variables()),
                model.constants,
                model.copula if model.copula is not None else model.correlation,
            )
        )
    except (TypeError, RecursionError):
        return None


def _check_form(form, model, limit_state):
    """Require a converged calculation for this unchanged problem."""
    if not form._results_valid or not form._converged:
        raise AnalysisError(
            "This method requires a successfully converged FORM analysis"
        )
    if (
        form.model is not model
        or form.limit_state.expression is not limit_state.expression
    ):
        raise ModelError(
            "Supplied FORM must use the same model and limit-state expression"
        )
    if (
        form._run_expression is not limit_state.expression
        or form._run_options != form.options
    ):
        raise AnalysisError(
            "FORM inputs have changed; rerun the supplied FORM analysis"
        )
    current = _problem_state(model)
    if current is None or form._run_model_state is None:
        form.run()
        if not form._converged:
            raise AnalysisError(
                "This method requires a successfully converged FORM analysis"
            )
    elif current != form._run_model_state:
        raise AnalysisError(
            "FORM model inputs have changed; rerun the supplied FORM analysis"
        )


def _check_coordinates(form, transform):
    """Reject design coordinates from another reference transformation."""
    for name in ("standard_space", "transform_type", "order", "T"):
        if _state(getattr(form.transform, name, None)) != _state(
            getattr(transform, name, None)
        ):
            raise ModelError(
                "Supplied FORM must use the same transformation and conditioning order"
            )
