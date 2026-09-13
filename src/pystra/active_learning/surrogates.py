"""Optional active-learning reliability in independent normal coordinates.

AK-MCS: Echard, Gayton and Lemaire (2011), doi:10.1016/j.strusafe.2011.01.002.
EFF: Bichon et al. (2008), doi:10.2514/1.34321.
Adaptive sparse bootstrap PCE: Marelli and Sudret (2018),
doi:10.1016/j.strusafe.2018.06.003, using UQLab 2.2.0 selection rules.
See THIRD_PARTY_NOTICES and docs/uqlab-pce-provenance.md.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from ._pce import _multi_indices, _hermite_basis, _fit_lars, _fit_ols
from ._validation import _positive_integer, _points, _training

__all__ = [
    "Surrogate",
    "KrigingSurrogate",
    "PCECandidate",
    "PCEFitResult",
    "EnsembleSurrogate",
    "PCESurrogate",
]


class Surrogate(ABC):
    """Predict a scalar limit state in independent standard normal space.

    Both methods use row-wise points, shape (n_samples, n_variables).
    Implementations must replace fitted state on every fit.
    """

    @abstractmethod
    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        """Fit finite scalar observations, shape (n_samples,)."""

    @abstractmethod
    def predict(self, points: np.ndarray) -> tuple:
        """Return finite mean and nonnegative std arrays, each (n_samples,)."""


class KrigingSurrogate(Surrogate):
    """Matérn 5/2 Gaussian process with normalized response.

    Parameters are keyword-only. ``noise`` is a positive numerical nugget;
    ``n_restarts`` controls additional likelihood optimization starts.
    ``seed`` owns the scikit-learn optimizer's random state.
    Requires ``pip install 'pystra[al]'``. Inputs are normal coordinates.
    """

    def __init__(
        self, *, n_restarts: int = 2, noise: float = 1e-10, seed: Optional[int] = None
    ) -> None:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        except ImportError as exc:
            raise ImportError(
                "Kriging requires scikit-learn; install with pip install 'pystra[al]'"
            ) from exc
        _positive_integer(n_restarts, "n_restarts", 0)
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("noise must be finite and positive")
        self._regressor = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e5)) * Matern(1.0, (1e-2, 1e4), nu=2.5),
            n_restarts_optimizer=n_restarts,
            alpha=noise,
            normalize_y=True,
            random_state=seed,
        )
        self._fitted = False

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        """Fit normal-coordinate observations; reject nonfinite inputs."""
        self._fitted = False
        points, values = _training(points, values)
        self._regressor.fit(points, values)
        self._fitted = True

    def predict(self, points: np.ndarray) -> tuple:
        """Return posterior mean/std; raise RuntimeError before fitting."""
        if not self._fitted:
            raise RuntimeError("Surrogate has not been fitted")
        return self._regressor.predict(_points(points), return_std=True)


@dataclass(frozen=True)
class PCECandidate:
    """A candidate degree/truncation and its final OLS error diagnostics."""

    degree: int
    q_norm: float
    n_candidates: int
    n_selected: int
    loo_error: float
    corrected_loo_error: float


@dataclass(frozen=True)
class PCEFitResult:
    """Immutable selected Hermite expansion and evaluated candidate scores.

    ``indices`` contains one power tuple per coefficient in normal-coordinate
    variable order. Errors are normalized by population variance of training
    responses. LOO is conditional on the chosen support, not a complete
    cross-validation of model selection or a failure-probability error bound.
    """

    degree: int
    q_norm: float
    indices: tuple
    coefficients: tuple
    loo_error: float
    corrected_loo_error: float
    candidates: tuple
    n_rank_deficient_bootstrap: int


class EnsembleSurrogate(Surrogate):
    """A surrogate that also exposes row-wise replicate predictions."""

    @abstractmethod
    def predict_replicates(self, points: np.ndarray) -> np.ndarray:
        """Return (n_points, n_replicates>=2) finite response predictions."""


class PCESurrogate(EnsembleSurrogate):
    """Adaptive sparse Hermite PCE with bootstrap local prediction spread.

    Parameters
    ----------
    degree : int or sequence of int
        Candidate maximum degrees, default (1, 2, 3, 4, 5). An integer fits
        a single candidate degree. Sequences must be strictly increasing.
    method : str
        'lars' (default) selects sparse supports using hybrid LARS and
        corrected leave-one-out error; 'ols' fits each complete candidate.
    q_norm : float or sequence of float
        Hyperbolic truncations in (0, 1], default 1. Sequences increase.
        A power tuple is admissible when sum(power**q) <= degree**q.
    max_interaction : int, optional
        Maximum number of variables in an interaction, default unrestricted.
    degree_early_stop, q_norm_early_stop : bool
        Default True. Stop degrees after two candidates below the best score;
        stop q-norms after two successive nonimprovements, as in UQLab 2.2.0.
        Set False for exhaustive searches, e.g. isolated higher-order terms.
    n_bootstrap : int
        Number of pairs-bootstrap refits on the selected support, default 30.
    max_terms : int
        Maximum candidate basis size (10000); excessive requests fail clearly.
    seed : int, optional
        Owns repeatable bootstrap resampling; no global RNG state is changed.

    Notes
    -----
    Adapted from UQLab 2.2.0, copyright Stefano Marelli and Bruno Sudret
    (ETH Zurich); see THIRD_PARTY_NOTICES. Candidate supports are selected
    on centered, normalized LAR paths, refitted by OLS and compared using
    corrected LOO. SVD replaces normal-equation inverses. The bootstrap keeps
    the selected sparse basis fixed (UQLab's fast bootstrap), so it does not
    quantify model-selection or truncation bias. Rank-deficient bootstrap
    draws use minimum-norm least squares and are counted in fit_result.
    No Gaussian error bound is implied.

    ``fit_result`` snapshots the chosen degree, q-norm, coefficients, indices
    and candidate diagnostics. Every fit starts a fresh adaptive search.
    Inputs and predictions use independent standard normal coordinates.
    """

    def __init__(
        self,
        *,
        degree: Union[int, Sequence[int]] = (1, 2, 3, 4, 5),
        method: str = "lars",
        q_norm: Union[float, Sequence[float]] = 1.0,
        max_interaction: Optional[int] = None,
        degree_early_stop: bool = True,
        q_norm_early_stop: bool = True,
        n_bootstrap: int = 30,
        max_terms: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        degrees = (degree,) if np.isscalar(degree) else tuple(degree)
        self.degree = tuple(_positive_integer(value, "degree") for value in degrees)
        if not self.degree or any(b <= a for a, b in zip(self.degree, self.degree[1:])):
            raise ValueError("degree must be nonempty and strictly increasing")
        norms = (q_norm,) if np.isscalar(q_norm) else tuple(q_norm)
        if not norms or any(
            isinstance(q, (bool, np.bool_)) or not np.isfinite(q) or not 0 < q <= 1
            for q in norms
        ):
            raise ValueError("q_norm must contain finite numbers in (0, 1]")
        self.q_norm = tuple(float(q) for q in norms)
        if any(b <= a for a, b in zip(self.q_norm, self.q_norm[1:])):
            raise ValueError("q_norm must be strictly increasing")
        if method not in ("lars", "ols"):
            raise ValueError("method must be 'lars' or 'ols'")
        self.method = method
        self.max_interaction = (
            None
            if max_interaction is None
            else _positive_integer(max_interaction, "max_interaction")
        )
        for name, value in (
            ("degree_early_stop", degree_early_stop),
            ("q_norm_early_stop", q_norm_early_stop),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
        self.degree_early_stop = degree_early_stop
        self.q_norm_early_stop = q_norm_early_stop
        self.n_bootstrap = _positive_integer(n_bootstrap, "n_bootstrap", 2)
        self.max_terms = _positive_integer(max_terms, "max_terms")
        self.seed = seed
        self._coefficients = None
        self.fit_result = None

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        """Select degree/truncation/support, then bootstrap its OLS fit."""
        self._coefficients = None
        self.fit_result = None
        points, values = _training(points, values)
        if len(points) < 3:
            raise ValueError("PCE requires at least three training points")
        dimension = points.shape[1]
        interaction = self.max_interaction or dimension
        best = None
        candidates = []
        degree_errors = []
        for degree in self.degree:
            q_errors = []
            previous_size = None
            for q_norm in self.q_norm:
                indices = _multi_indices(
                    dimension, degree, q_norm, interaction, self.max_terms
                )
                if len(indices) == previous_size:
                    continue
                previous_size = len(indices)
                basis = _hermite_basis(points, indices)
                try:
                    if self.method == "lars":
                        support, regression = _fit_lars(basis, values)
                    else:
                        support = np.arange(len(indices))
                        regression = _fit_ols(basis, values)
                except ValueError:
                    # Unidentifiable dense candidates cannot win selection.
                    if len(self.degree) * len(self.q_norm) == 1:
                        raise
                    candidates.append(
                        PCECandidate(degree, q_norm, len(indices), 0, np.inf, np.inf)
                    )
                    q_errors.append(np.inf)
                    continue
                candidate = PCECandidate(
                    degree,
                    q_norm,
                    len(indices),
                    len(support),
                    regression.loo,
                    regression.corrected_loo,
                )
                candidates.append(candidate)
                q_errors.append(candidate.corrected_loo_error)
                if np.isfinite(candidate.corrected_loo_error) and (
                    best is None
                    or candidate.corrected_loo_error < best[0].corrected_loo_error
                ):
                    best = (
                        candidate,
                        indices[support],
                        regression.coefficients,
                        basis[:, support],
                    )
                finite_q_errors = [error for error in q_errors if np.isfinite(error)]
                if (
                    self.q_norm_early_stop
                    and len(finite_q_errors) >= 3
                    and finite_q_errors[-1]
                    >= finite_q_errors[-2]
                    >= finite_q_errors[-3]
                ):
                    break
            degree_error = min(q_errors, default=np.inf)
            if np.isfinite(degree_error):
                degree_errors.append(degree_error)
            if (
                best is not None
                and self.degree_early_stop
                and len(degree_errors) >= 3
                and all(
                    error > best[0].corrected_loo_error for error in degree_errors[-2:]
                )
            ):
                break
        if best is None:
            raise ValueError(
                "No identifiable PCE candidate; increase training size or reduce degree"
            )
        candidate, indices, coefficients, basis = best
        rng = np.random.default_rng(self.seed)
        ensemble = []
        rank_deficient = 0
        for _ in range(self.n_bootstrap):
            resample = rng.integers(len(points), size=len(points))
            fitted, _, rank, _ = np.linalg.lstsq(
                basis[resample], values[resample], rcond=None
            )
            rank_deficient += rank < basis.shape[1]
            if not np.all(np.isfinite(fitted)):
                raise ValueError("Nonfinite PCE bootstrap coefficients")
            ensemble.append(fitted)
        self._dimension = dimension
        self._powers = indices
        self._ensemble = np.asarray(ensemble).T
        self._coefficients = coefficients
        self.fit_result = PCEFitResult(
            candidate.degree,
            candidate.q_norm,
            tuple(tuple(int(power) for power in row) for row in indices),
            tuple(float(coefficient) for coefficient in coefficients),
            candidate.loo_error,
            candidate.corrected_loo_error,
            tuple(candidates),
            int(rank_deficient),
        )

    def predict(self, points: np.ndarray) -> tuple:
        """Return full-design mean and selected-basis bootstrap spread."""
        if self._coefficients is None:
            raise RuntimeError("Surrogate has not been fitted")
        points = _points(points)
        if points.shape[1] != self._dimension:
            raise ValueError("Prediction dimension differs from training dimension")
        basis = _hermite_basis(points, self._powers)
        return basis @ self._coefficients, np.std(
            basis @ self._ensemble, axis=1, ddof=1
        )

    def predict_replicates(self, points: np.ndarray) -> np.ndarray:
        """Return individual selected-support bootstrap responses (columns).

        Replicate columns refer to the same fitted coefficients across calls,
        enabling probability estimates on a common sample. They do not include
        model-selection uncertainty. Returned arrays do not alias fitted state.
        """
        if self._coefficients is None:
            raise RuntimeError("Surrogate has not been fitted")
        points = _points(points)
        if points.shape[1] != self._dimension:
            raise ValueError("Prediction dimension differs from training dimension")
        return _hermite_basis(points, self._powers) @ self._ensemble
