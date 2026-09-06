"""Immutable records of reliability analyses, independent of solver state."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["FormResult"]


@dataclass(frozen=True)
class FormResult:
    """Snapshot returned by FORM.

    ``beta`` is normal-equivalent; ``geometric_beta`` is the signed distance
    in ``standard_space``. Points and directions are immutable one-dimensional
    tuples in ``variable_names`` order. Failed analyses have no probability,
    index or design point; their convergence diagnostics remain available.
    """

    converged: bool
    beta: Optional[float]
    failure_probability: Optional[float]
    geometric_beta: Optional[float]
    variable_names: Tuple[str, ...]
    design_point: Optional[Tuple[float, ...]]
    standard_point: Optional[Tuple[float, ...]]
    alpha: Optional[Tuple[float, ...]]
    standard_space: str
    iterations: int
    limit_state_error: Optional[float]
    direction_error: Optional[float]
    message: str

    @classmethod
    def from_analysis(cls, analysis) -> "FormResult":
        """Copy a completed solver's numerical results and diagnostics."""
        valid = bool(analysis.converged and analysis.results_valid)
        standard_space = getattr(analysis.transform, "standard_space", "normal")
        vector = lambda value: tuple(float(x) for x in np.asarray(value).ravel())
        return cls(
            converged=valid,
            beta=(
                (
                    float(analysis.beta)
                    if standard_space == "normal"
                    else analysis.get_equivalent_beta()
                )
                if valid
                else None
            ),
            failure_probability=float(analysis.Pf) if valid else None,
            geometric_beta=float(analysis.beta) if valid else None,
            variable_names=tuple(analysis.model.get_variables()),
            design_point=vector(analysis.get_design_point(False)) if valid else None,
            standard_point=vector(analysis.u) if valid else None,
            alpha=vector(analysis.alpha) if valid else None,
            standard_space=standard_space,
            iterations=analysis.i or 0,
            limit_state_error=analysis.e1,
            direction_error=analysis.e2,
            message="Converged" if valid else "FORM iteration limit reached",
        )
