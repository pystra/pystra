"""Tabular reliability summaries shared by engineering workflows."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import norm

__all__ = ["reliability_row"]


def _probabilities(pf: Any, beta: Any) -> tuple[float, float]:
    """Read probability/index pairs without losing a supplied tail index."""
    if pf is None and beta is None:
        raise ValueError("analysis result must provide pf and/or beta")
    if pf is not None:
        pf = float(pf)
        if not np.isfinite(pf) or not 0 <= pf <= 1:
            raise ValueError("failure_probability must be finite and in [0, 1]")
    if beta is not None:
        beta = float(beta)
        if np.isnan(beta):
            raise ValueError("beta must not be NaN for a successful analysis")
    if pf is None:
        pf = float(norm.cdf(-beta))
    if beta is None:
        beta = -float(norm.ppf(pf))
    return pf, beta


def reliability_row(
    result: Any, *, probability_name: str = "failure_probability"
) -> dict:
    """Return probabilities and explicit analysis status for a table row.

    Parameters
    ----------
    result : reliability result, mapping, sequence or float
        Records retain their method, status and message. For an analytic
        callback, a probability, ``(pf, beta)`` pair or mapping with ``pf``
        and/or ``beta`` is also accepted; without status it denotes a completed
        evaluation. Failed records retain their diagnostics, with NaN table
        estimates so downstream calculations cannot reuse an untrusted value.
    probability_name : str, default "failure_probability"
        Column name; decision tables use ``pf``.

    Returns
    -------
    dict
        Probability, beta, converged, status, message and method. Probability
        and beta are dimensionless, for the input event and reference period.
        No reference-period conversion or independence assumption is made.
    """
    if isinstance(result, Mapping):
        read = result.get
        pf = read("pf", read("failure_probability", read("failure")))
        beta = read("beta", read("reliability_index"))
    elif hasattr(result, "failure_probability") or hasattr(result, "beta"):
        read = lambda name, default=None: getattr(result, name, default)
        pf, beta = read("failure_probability"), read("beta")
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        read = lambda name, default=None: default
        pf = result[0] if len(result) else None
        beta = result[1] if len(result) > 1 else None
    else:
        read = lambda name, default=None: default
        pf, beta = result, None
    status = read("status")
    converged = read("converged")
    if status is None:
        status = (
            "not_converged" if converged is not None and not converged else "completed"
        )
    if converged is None:
        converged = status in ("converged", "completed")
    if not isinstance(converged, (bool, np.bool_)):
        raise TypeError("converged must be a boolean")
    if bool(converged) != (status in ("converged", "completed")):
        raise ValueError("Analysis status and convergence flag disagree")
    pf, beta = _probabilities(pf, beta) if converged else (np.nan, np.nan)
    return {
        probability_name: pf,
        "beta": beta,
        "converged": bool(converged),
        "status": status,
        "message": str(read("message", "")),
        "method": str(read("method", "callback")),
    }
