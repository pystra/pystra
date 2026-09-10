"""Numerical sparse-PCE kernels adapted from UQLab 2.2.0.

Copyright (c) 2018-2026, Stefano Marelli and Bruno Sudret (ETH Zurich).
Adapted for PySTRA in 2026; see THIRD_PARTY_NOTICES for the BSD terms,
and docs/uqlab-pce-provenance.md for source routines and differences.
"""

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.special import eval_hermitenorm


@dataclass
class _Regression:
    coefficients: np.ndarray
    loo: float
    corrected_loo: float


def _multi_indices(dimension, degree, q_norm, max_interaction, max_terms):
    # Generate only admissible branches, rather than materializing the full
    # total-degree basis before hyperbolic/interaction truncation.
    indices = [np.zeros(dimension, dtype=int)]
    frontier = [()]
    for _ in range(degree):
        following = []
        for previous in frontier:
            for axis in range(previous[-1] if previous else 0, dimension):
                candidate = previous + (axis,)
                powers = np.bincount(candidate, minlength=dimension)
                if np.count_nonzero(powers) > max_interaction:
                    continue
                if np.sum(powers**q_norm) > degree**q_norm + 1e-12:
                    continue
                indices.append(powers)
                following.append(candidate)
                if len(indices) > max_terms:
                    raise ValueError(
                        "PCE candidate basis exceeds max_terms; reduce degree, q_norm or max_interaction"
                    )
        frontier = following
    indices.sort(
        key=lambda row: (
            sum(row),
            np.count_nonzero(row),
            tuple(sorted(row[row > 0], reverse=True)),
            tuple(row),
        )
    )
    return np.asarray(indices)


def _hermite_basis(points, indices):
    basis = np.ones((len(points), len(indices)))
    for axis in range(points.shape[1]):
        for power in np.unique(indices[:, axis]):
            if power:
                values = eval_hermitenorm(int(power), points[:, axis]) / np.sqrt(
                    float(factorial(int(power)))
                )
                basis[:, indices[:, axis] == power] *= values[:, None]
    if not np.all(np.isfinite(basis)):
        raise ValueError("PCE polynomial evaluations are nonfinite")
    return basis


def _fit_ols(basis, values, *, intercept_leverage=0.0):
    """SVD least squares and UQLab's ordinary/corrected PRESS estimates.

    For centered LARS path scoring, intercept leverage is 1/N, while the
    correction counts the active nonconstant regressors. Final model scoring
    uses the uncentered basis, including its constant column, as in UQLab.
    """
    count, terms = basis.shape
    if count <= terms:
        raise ValueError("PCE requires more training points than selected basis terms")
    left, singular, right = np.linalg.svd(basis, full_matrices=False)
    if (
        not len(singular)
        or singular[-1] <= singular[0] * max(basis.shape) * np.finfo(float).eps
    ):
        raise ValueError("PCE training design is rank deficient")
    coefficients = right.T @ ((left.T @ values) / singular)
    variance = np.var(values)
    if variance == 0:
        return _Regression(coefficients, 0.0, 0.0)
    leverage = np.sum(left**2, axis=1) + intercept_leverage
    denominator = 1 - leverage
    if np.any(denominator <= 10 * np.finfo(float).eps):
        return _Regression(coefficients, np.inf, np.inf)
    residual = basis @ coefficients - values
    roundoff = (
        10
        * np.finfo(float).eps
        * (
            np.linalg.norm(values)
            + np.linalg.norm(basis) * np.linalg.norm(coefficients)
        )
    )
    loo = (
        0.0
        if np.linalg.norm(residual) <= roundoff
        else float(np.mean((residual / denominator) ** 2) / variance)
    )
    correction = count / (count - terms) * (1 + np.sum(singular**-2))
    return _Regression(coefficients, loo, float(loo * correction))


def _fit_lars(basis, values):
    """Select a LAR path support by centered corrected LOO, then refit OLS."""
    count, terms = basis.shape
    selected = np.array([0])
    best = _fit_ols(basis[:, selected], values)
    centered_values = values - values.mean()
    if not np.any(centered_values) or terms == 1:
        return selected, best
    centered = basis[:, 1:] - basis[:, 1:].mean(axis=0)
    scales = np.std(centered, axis=0, ddof=1)
    usable = np.flatnonzero(scales > np.finfo(float).eps)
    if not len(usable):
        return selected, best
    normalized = centered[:, usable] / scales[usable]
    best_score = best.corrected_loo
    active = []
    inactive = list(range(len(usable)))
    direction_mean = np.zeros(count)
    coefficients = np.zeros(len(usable))
    maximum = min(count - 2, len(usable))
    for step in range(maximum):
        correlations = normalized.T @ (centered_values - direction_mean)
        chosen = inactive[int(np.argmax(np.abs(correlations[inactive])))]
        largest = abs(correlations[chosen])
        active.append(chosen)
        active_basis = normalized[:, active]
        _, singular, right = np.linalg.svd(active_basis, full_matrices=False)
        if singular[-1] <= singular[0] * max(active_basis.shape) * np.finfo(float).eps:
            active.pop()
            inactive.remove(chosen)
            continue
        inverse = (right.T / singular**2) @ right
        signs = np.sign(correlations[active])
        signs[signs == 0] = 1
        scale = 1 / np.sqrt(signs @ inverse @ signs)
        weights = scale * inverse @ signs
        direction = active_basis @ weights
        angles = normalized.T @ direction
        if step < maximum - 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                distances = np.r_[
                    (largest - correlations[inactive]) / (scale - angles[inactive]),
                    (largest + correlations[inactive]) / (scale + angles[inactive]),
                ]
            positive = distances[distances > 0]
            distance = min(positive) if len(positive) else 0.0
        else:
            distance = largest / scale
        inactive.remove(chosen)
        direction_mean += distance * direction
        coefficients[active] += distance * weights
        try:
            fitted = _fit_ols(
                active_basis, centered_values, intercept_leverage=1 / count
            )
        except ValueError:
            continue
        if fitted.corrected_loo < best_score:
            best_score = fitted.corrected_loo
            selected = np.r_[0, usable[np.flatnonzero(coefficients)] + 1]
    # Normalized/centered path errors choose support; unscaled errors compare
    # degrees and q-norms. These two criteria intentionally differ in UQLab.
    return selected, _fit_ols(basis[:, selected], values)
