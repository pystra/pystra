"""Shared validation for active-learning numerical boundaries."""

import numpy as np


def _positive_integer(value, name, minimum=1):
    """Return an integer meeting the lower bound; reject boolean values."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _points(points):
    """Return a nonempty finite array with rows of points and columns of variables."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or 0 in points.shape or not np.all(np.isfinite(points)):
        raise ValueError(
            "points must be a nonempty finite (n_samples, n_variables) array"
        )
    return points


def _training(points, values):
    """Validate training rows and one finite response per row, preserving their order."""
    points = _points(points)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(points),) or not np.all(np.isfinite(values)):
        raise ValueError("values must be finite with shape (n_samples,)")
    return points, values


def _predictions(mean, std):
    """Validate matching finite prediction vectors with nonnegative standard deviations."""
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if mean.ndim != 1 or not mean.size or mean.shape != std.shape:
        raise ValueError(
            "mean and std must have the same nonempty one-dimensional shape"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0):
        raise ValueError("Predictions must be finite and std nonnegative")
    return mean, std
