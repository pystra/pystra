"""Sequential PC-Kriging with a selected Hermite trend and universal variance.

Schöbi, Sudret & Wiart (2015), doi:10.1615/Int.J.UncertaintyQuantification.2015012467. See also
Schöbi, Sudret & Marelli, doi:10.1061/AJRUA6.0000870.
UQLab supplied the sparse-PCE basis and a comparison for universal prediction;
see THIRD_PARTY_NOTICES and docs/active-learning-provenance.md.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from ._pce import _hermite_basis
from ._validation import _positive_integer, _points, _training
from .surrogates import PCESurrogate, PCEFitResult, Surrogate

__all__ = ["PCKrigingFitResult", "PCKrigingSurrogate"]


@dataclass(frozen=True)
class PCKrigingFitResult:
    """Snapshot conditional on the selected trend and fitted kernel.

    ``trend`` records the preliminary sparse PCE; ``coefficients`` are the
    subsequent GLS trend coefficients in physical response units. Length
    scales follow input column order. Predictive variance does not integrate
    uncertainty in basis selection or kernel hyperparameters. ``optimized``
    records whether the length scales were fitted. ``objective`` is the
    profiled likelihood criterion per observation for the scaled response,
    with additive constants omitted; it is not a reliability error metric.
    """

    trend: PCEFitResult
    coefficients: tuple
    length_scale: tuple
    process_variance: float
    optimized: bool
    objective: float


class PCKrigingSurrogate(Surrogate):
    """Sequential sparse PC-Kriging in independent normal coordinates.

    Parameters
    ----------
    degree : int or sequence of int
        Sparse Hermite trend candidates; default (1, 2, 3).
    q_norm : float
        Hyperbolic truncation, default 1.
    max_interaction : int, optional
        Maximum polynomial interaction order.
    correlation : str
        'matern52' (default, consistent with KrigingSurrogate) or 'gaussian'.
    length_scale : float or sequence of float
        Initial correlation scales, default 1; a scalar is broadcast
        across dimensions. Fixed scales are possible with ``optimize=False``.
    optimize : bool
        Default True: fit anisotropic scales by profiled maximum likelihood.
        Bounds are [0.01, 100] in standard-normal coordinate units.
    noise : float
        Positive relative diagonal nugget, default 1e-8. Predictions describe
        the latent response, excluding new observation noise.
    n_restarts : int
        Additional random optimizer starts, default 0. Failed optimization
        raises RuntimeError; it cannot silently produce a successful fit.
    seed : int, optional
        Local generator seed for additional optimization starts.

    Notes
    -----
    The sparse trend is selected before optimizing the correlation model;
    this is sequential PC-Kriging, not optimal PC-Kriging's search along every
    LAR support. GLS re-estimates the trend for each correlation candidate.
    Universal predictive variance includes trend-estimation uncertainty.
    Exactly representable training data yield zero residual process variance;
    this does not certify correctness outside the observations. Requires only
    NumPy/SciPy, with row-wise input points and scalar responses.
    """

    def __init__(
        self,
        *,
        degree: int | Sequence[int] = (1, 2, 3),
        q_norm: float = 1.0,
        max_interaction: int | None = None,
        length_scale: float | Sequence[float] = 1.0,
        correlation: str = "matern52",
        optimize: bool = True,
        noise: float = 1e-8,
        n_restarts: int = 0,
        seed: int | None = None,
    ) -> None:
        self._trend = PCESurrogate(
            degree=degree,
            q_norm=q_norm,
            max_interaction=max_interaction,
            n_bootstrap=2,
            seed=seed,
        )
        scales = np.atleast_1d(np.asarray(length_scale, dtype=float))
        if (
            scales.ndim != 1
            or not len(scales)
            or np.any(~np.isfinite(scales))
            or np.any(scales <= 0)
        ):
            raise ValueError("length_scale must contain finite positive values")
        if not isinstance(optimize, bool):
            raise ValueError("optimize must be a bool")
        if optimize and np.any((scales < 0.01) | (scales > 100)):
            raise ValueError("Optimized initial length_scale must be in [0.01, 100]")
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("noise must be finite and positive")
        if correlation not in ("gaussian", "matern52"):
            raise ValueError("correlation must be 'gaussian' or 'matern52'")
        self.correlation = correlation
        self.length_scale = tuple(scales)
        self.optimize = optimize
        self.noise = float(noise)
        self.n_restarts = _positive_integer(n_restarts, "n_restarts", 0)
        self.seed = seed
        self.fit_result = None

    def _correlation(self, first, second, scales):
        squared = cdist(first / scales, second / scales, "sqeuclidean")
        if self.correlation == "gaussian":
            return np.exp(-0.5 * squared)
        distance = np.sqrt(5 * squared)
        return (1 + distance + distance**2 / 3) * np.exp(-distance)

    def _solve(self, scales, points, basis, values):
        correlation = self._correlation(points, points, scales)
        correlation.flat[:: len(points) + 1] += self.noise
        factor = cholesky(correlation, lower=True)
        whitened_basis = solve_triangular(factor, basis, lower=True)
        whitened_values = solve_triangular(factor, values, lower=True)
        orthogonal, triangular = np.linalg.qr(whitened_basis, mode="reduced")
        if np.linalg.matrix_rank(triangular) < basis.shape[1]:
            raise ValueError("PC-Kriging trend is rank deficient")
        coefficients = solve_triangular(triangular, orthogonal.T @ whitened_values)
        residual = whitened_values - whitened_basis @ coefficients
        variance = float(residual @ residual / len(points))
        objective = (
            len(points) * np.log(max(variance, 1e-30))
            + 2 * np.log(np.diag(factor)).sum()
        )
        return (
            objective,
            factor,
            whitened_basis,
            triangular,
            coefficients,
            residual,
            variance,
        )

    def _objective(self, log_scales, points, basis, values):
        scales = np.exp(log_scales)
        objective, factor, _, _, _, residual, variance = self._solve(
            scales, points, basis, values
        )
        inverse = cho_solve((factor, True), np.eye(len(points)))
        alpha = solve_triangular(factor.T, residual)
        sensitivity = inverse - np.outer(alpha, alpha) / max(variance, 1e-30)
        squared = ((points[:, None, :] - points[None, :, :]) / scales) ** 2
        if self.correlation == "gaussian":
            derivative = np.exp(-0.5 * np.sum(squared, axis=2))
        else:
            distance = np.sqrt(5 * np.sum(squared, axis=2))
            derivative = (5 / 3) * (1 + distance) * np.exp(-distance)
        gradient = np.einsum("ij,ij,ijk->k", sensitivity, derivative, squared)
        return objective / len(points), gradient / len(points)

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        """Replace fitted state; errors leave prediction unavailable."""
        self.fit_result = None
        points, values = _training(points, values)
        dimension = points.shape[1]
        if len(self.length_scale) not in (1, dimension):
            raise ValueError("length_scale must be scalar or match input dimension")
        scales = np.broadcast_to(self.length_scale, (dimension,)).copy()
        self._trend.fit(points, values)
        powers = np.asarray(self._trend.fit_result.indices)
        basis = _hermite_basis(points, powers)
        # Scaling preserves the physical Hermite intercept and GLS solution.
        scale = max(
            float(np.std(values)),
            float(np.max(np.abs(values))) * np.finfo(float).eps,
            np.finfo(float).tiny,
        )
        normalized = values / scale
        polynomial = basis @ np.asarray(self._trend.fit_result.coefficients)
        exact = np.linalg.norm(values - polynomial) <= 100 * np.finfo(float).eps * max(
            np.linalg.norm(values), np.finfo(float).tiny
        )
        if self.optimize and not exact:

            def objective(log_scales):
                return self._objective(log_scales, points, basis, normalized)

            rng = np.random.default_rng(self.seed)
            starts = [np.log(scales)] + [
                rng.uniform(np.log(0.01), np.log(100), dimension)
                for _ in range(self.n_restarts)
            ]
            fits = [
                minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(np.log(0.01), np.log(100))] * dimension,
                    options={"maxiter": 200, "ftol": 1e-9, "maxls": 40},
                )
                for start in starts
            ]
            successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
            if not successful:
                raise RuntimeError(
                    "PC-Kriging likelihood optimization did not converge"
                )
            scales = np.exp(min(successful, key=lambda fit: fit.fun).x)
        (
            objective,
            factor,
            whitened_basis,
            triangular,
            coefficients,
            residual,
            variance,
        ) = self._solve(scales, points, basis, normalized)
        if exact:
            variance = 0.0
        self._points = points.copy()
        self._powers = powers
        self._scales = scales
        self._factor = factor
        self._whitened_basis = whitened_basis
        self._triangular = triangular
        self._coefficients = coefficients * scale
        self._alpha = solve_triangular(factor.T, residual) * scale
        self._variance = variance * scale**2
        self.fit_result = PCKrigingFitResult(
            self._trend.fit_result,
            tuple(float(value) for value in self._coefficients),
            tuple(float(value) for value in scales),
            self._variance,
            self.optimize and not exact,
            float(objective / len(points)),
        )

    def predict(self, points: np.ndarray) -> tuple:
        """Return universal mean/std conditional on fitted trend and scales."""
        if self.fit_result is None:
            raise RuntimeError("Surrogate has not been fitted")
        points = _points(points)
        if points.shape[1] != self._points.shape[1]:
            raise ValueError("Prediction dimension differs from training dimension")
        basis = _hermite_basis(points, self._powers)
        cross = self._correlation(points, self._points, self._scales)
        projected = solve_triangular(self._factor, cross.T, lower=True)
        mismatch = basis.T - self._whitened_basis.T @ projected
        correction = solve_triangular(self._triangular.T, mismatch, lower=True)
        relative_variance = (
            1 - np.sum(projected**2, axis=0) + np.sum(correction**2, axis=0)
        )
        if np.any(relative_variance < -1e-8):
            raise FloatingPointError(
                "PC-Kriging predictive variance is materially negative"
            )
        return basis @ self._coefficients + cross @ self._alpha, np.sqrt(
            self._variance * np.maximum(relative_variance, 0)
        )
