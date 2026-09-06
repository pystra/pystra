"""Optional active-learning reliability in independent normal coordinates.

AK-MCS: Echard, Gayton and Lemaire (2011), doi:10.1016/j.strusafe.2011.01.002.
EFF: Bichon et al. (2008), doi:10.2514/1.34321.
Adaptive sparse bootstrap PCE: Marelli and Sudret (2018),
doi:10.1016/j.strusafe.2018.06.003, using UQLab 2.2.0 selection rules.
See THIRD_PARTY_NOTICES and docs/uqlab-pce-provenance.md.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import comb
from typing import Optional, Sequence, Union
import warnings

import numpy as np
from scipy.stats import norm, qmc

from .analysis import AnalysisObject
from ._pce import _multi_indices, _hermite_basis, _fit_lars, _fit_ols

__all__ = [
    "ActiveLearning",
    "ActiveLearningResult",
    "LearningStep",
    "Surrogate",
    "KrigingSurrogate",
    "PceSurrogate",
    "PceFitResult",
    "PceCandidate",
    "learning_u",
    "learning_eff",
]


def _positive_integer(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or 0 in points.shape or not np.all(np.isfinite(points)):
        raise ValueError(
            "points must be a nonempty finite (n_samples, n_variables) array"
        )
    return points


def _training(points, values):
    points = _points(points)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(points),) or not np.all(np.isfinite(values)):
        raise ValueError("values must be finite with shape (n_samples,)")
    return points, values


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
    ):
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
class PceCandidate:
    """A candidate degree/truncation and its final OLS error diagnostics."""

    degree: int
    q_norm: float
    n_candidates: int
    n_selected: int
    loo_error: float
    corrected_loo_error: float


@dataclass(frozen=True)
class PceFitResult:
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


class PceSurrogate(Surrogate):
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
    ):
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
                        PceCandidate(degree, q_norm, len(indices), 0, np.inf, np.inf)
                    )
                    q_errors.append(np.inf)
                    continue
                candidate = PceCandidate(
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
        self.fit_result = PceFitResult(
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


def _predictions(mean, std):
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if mean.ndim != 1 or not mean.size or mean.shape != std.shape:
        raise ValueError(
            "mean and std must have the same nonempty one-dimensional shape"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0):
        raise ValueError("Predictions must be finite and std nonnegative")
    return mean, std


def learning_u(mean: np.ndarray, std: np.ndarray, *, threshold: float = 2.0) -> tuple:
    """Return the U score, best index (minimum), and threshold satisfaction.

    Zero spread gives infinity away from zero and zero at the boundary.
    Arrays have shape (n_candidates,). ``threshold`` must be positive.
    """
    mean, std = _predictions(mean, std)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    values = np.full_like(mean, np.inf)
    np.divide(np.abs(mean), std, out=values, where=std > 0)
    values[(std == 0) & (mean == 0)] = 0
    best = int(np.argmin(values))
    return values, best, bool(values[best] >= threshold)


def learning_eff(
    mean: np.ndarray, std: np.ndarray, *, threshold: float = 1e-3
) -> tuple:
    r"""Return expected feasibility, best index (maximum), and stopping flag.

    Here G is Gaussian with the supplied mean/std, each (n_candidates,).
    EFF is symmetric in mean and has the units of the limit state.
    ``threshold`` is an absolute, positive tolerance in those units.
    Zero spread gives zero EFF, including at the boundary.
    """
    mean, std = _predictions(mean, std)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    ratio = np.zeros_like(mean)
    with np.errstate(over="ignore"):
        np.divide(np.abs(mean), std, out=ratio, where=std > 0)
    ratio = np.minimum(ratio, 40.0)  # Gaussian payoff underflows beyond this tail
    # Integrate the two linear halves of the triangular feasibility function.
    lower, middle, upper = -2 - ratio, -ratio, 2 - ratio
    values = std * (
        (2 + ratio) * (norm.cdf(middle) - norm.cdf(lower))
        + (2 - ratio) * (norm.cdf(upper) - norm.cdf(middle))
        + norm.pdf(lower)
        + norm.pdf(upper)
        - 2 * norm.pdf(middle)
    )
    values = np.maximum(values, 0.0)  # roundoff only, not an absolute value
    best = int(np.argmax(values))
    return values, best, bool(values[best] < threshold)


@dataclass(frozen=True)
class LearningStep:
    """One fit's candidate probability, best learning score and true call count."""

    failure_probability: float
    learning_score: float
    n_evaluations: int


@dataclass(frozen=True)
class ActiveLearningResult:
    """Snapshot from a run, including unfinished estimates and stopping status.

    ``sampling_cov`` and the exact 95% binomial ``sampling_interval`` concern
    only the independent final population, conditional on the fitted surrogate.
    Neither includes surrogate bias. ``converged`` means the finite candidate
    learning criterion and sampling precision were met; it is not an accuracy
    guarantee. Zero/all failures never satisfy the sampling criterion.
    """

    failure_probability: float
    beta: float
    sampling_cov: float
    sampling_interval: tuple
    converged: bool
    status: str
    n_evaluations: int
    n_estimation: int
    history: tuple


class ActiveLearning(AnalysisObject):
    """Enrich a surrogate in independent normal coordinates, then estimate Pf.

    Parameters
    ----------
    stochastic_model, limit_state, analysis_options : optional
        Standard PySTRA analysis inputs. Failure is g <= 0.
    surrogate : str or Surrogate
        'kriging' (default), 'pce', or an explicitly supplied fitted-model
        implementation. A supplied object is refitted in place by run().
    learning_function : str
        'u' (default) or 'eff'. EFF assumes Gaussian predictive uncertainty.
    n_initial : int, optional
        Initial LHS size: max(12, 2*n_variables), or max(30, 5*n_variables)
        for sparse PCE. Dense OLS uses at least twice the largest total-degree
        basis size.
    n_candidates, n_estimation : int
        Enrichment pool (10000) and independent final MC population (100000).
    max_iterations : int
        Maximum added true evaluations (200), excluding the initial design.
    learning_threshold : float, optional
        U minimum (2), or EFF maximum (1e-3, in limit-state units).
    target_cov : float
        Required final conditional sampling CoV (0.1).
    surrogate_kwargs : mapping, optional
        Constructor settings for a named surrogate.
    seed : int, optional
        Repeatable run-owned LHS, candidates, final population and optimizer.

    Notes
    -----
    Candidate points are evaluated at most once. Predictions are batched.
    Nonconvergence warns and returns an explicit unfinished result. Exceptions
    invalidate any prior result. A finite pool may miss disconnected failures;
    bootstrap spread can miss common polynomial bias. Validate the surrogate
    independently for the problem being assessed.
    """

    def __init__(
        self,
        *,
        stochastic_model=None,
        limit_state=None,
        analysis_options=None,
        surrogate: Union[str, Surrogate] = "kriging",
        learning_function: str = "u",
        n_initial: Optional[int] = None,
        n_candidates: int = 10_000,
        n_estimation: int = 100_000,
        max_iterations: int = 200,
        learning_threshold: Optional[float] = None,
        target_cov: float = 0.1,
        surrogate_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            stochastic_model=stochastic_model,
            limit_state=limit_state,
            analysis_options=analysis_options,
        )
        if not isinstance(surrogate, Surrogate) and surrogate not in ("kriging", "pce"):
            raise ValueError("Unknown surrogate; use 'kriging', 'pce', or a Surrogate")
        if learning_function not in ("u", "eff"):
            raise ValueError("Unknown learning_function; use 'u' or 'eff'")
        if isinstance(surrogate, Surrogate) and surrogate_kwargs:
            raise ValueError("surrogate_kwargs requires a named surrogate")
        self.surrogate = surrogate
        self.learning_function = learning_function
        self.n_initial = (
            None if n_initial is None else _positive_integer(n_initial, "n_initial", 2)
        )
        self.n_candidates = _positive_integer(n_candidates, "n_candidates")
        self.n_estimation = _positive_integer(n_estimation, "n_estimation", 2)
        self.max_iterations = _positive_integer(max_iterations, "max_iterations", 0)
        self.learning_threshold = (
            learning_threshold
            if learning_threshold is not None
            else (2.0 if learning_function == "u" else 1e-3)
        )
        for name, value in (
            ("learning_threshold", self.learning_threshold),
            ("target_cov", target_cov),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.target_cov = target_cov
        self.surrogate_kwargs = dict(surrogate_kwargs or {})
        self.seed = seed
        self.result = None
        self.surrogate_model = None

    def _evaluate(self, points):
        marginals = self.model.marginal_distributions
        physical = np.asarray(
            [self.transform.u_to_x(point, marginals) for point in points]
        )
        values, _ = self.limitstate.evaluate_lsf(
            physical.T, self.model, self.options, "no"
        )
        return _training(points, np.asarray(values).ravel())[1]

    def _predict(self, surrogate, points):
        means, spreads = [], []
        for start in range(0, len(points), 2048):
            batch = points[start : start + 2048]
            mean, std = _predictions(*surrogate.predict(batch))
            if len(mean) != len(batch):
                raise ValueError("Surrogate returned the wrong number of predictions")
            means.append(mean)
            spreads.append(std)
        return np.concatenate(means), np.concatenate(spreads)

    def run(self) -> ActiveLearningResult:
        """Run from fresh state and return an immutable convergence record."""
        from scipy.stats import binomtest

        self.result = None
        self.surrogate_model = None
        self.results_valid = False
        self.init_run()
        dimension = self.model.n_marg
        if dimension < 1:
            raise ValueError("Active learning requires at least one random variable")
        rng = np.random.default_rng(self.seed)
        settings = dict(self.surrogate_kwargs)
        settings.setdefault("seed", int(rng.integers(2**31 - 1)))
        surrogate = self.surrogate
        if isinstance(surrogate, str):
            surrogate = {"kriging": KrigingSurrogate, "pce": PceSurrogate}[surrogate](
                **settings
            )
        initial = max(12, 2 * dimension)
        if isinstance(surrogate, PceSurrogate):
            if surrogate.method == "ols":
                maximum_degree = max(surrogate.degree)
                initial = max(
                    initial, 2 * comb(dimension + maximum_degree, maximum_degree)
                )
            else:
                initial = max(30, 5 * dimension)
        initial = self.n_initial or initial
        uniforms = qmc.LatinHypercube(
            dimension, seed=int(rng.integers(2**31 - 1))
        ).random(initial)
        design = norm.ppf(
            np.clip(uniforms, np.finfo(float).eps, 1 - np.finfo(float).eps)
        )
        observations = self._evaluate(design)
        candidates = rng.standard_normal((self.n_candidates, dimension))
        available = np.ones(self.n_candidates, dtype=bool)
        history = []
        learned = False
        status = "max_iterations"
        learning = learning_u if self.learning_function == "u" else learning_eff
        for iteration in range(self.max_iterations + 1):
            surrogate.fit(design, observations)
            mean, std = self._predict(surrogate, candidates)
            indices = np.flatnonzero(available)
            if not len(indices):
                status = "candidate_exhaustion"
                break
            scores, best, learned = learning(
                mean[indices], std[indices], threshold=self.learning_threshold
            )
            history.append(
                LearningStep(
                    float(np.mean(mean <= 0)), float(scores[best]), len(design)
                )
            )
            learned = learned and bool(np.any(mean <= 0) and np.any(mean > 0))
            if learned or iteration == self.max_iterations:
                break
            selected = indices[best]
            point = candidates[selected : selected + 1]
            observations = np.concatenate((observations, self._evaluate(point)))
            design = np.vstack((design, point))
            available[np.all(candidates == point, axis=1)] = False
        # This population never participates in fitting or enrichment.
        estimation = rng.standard_normal((self.n_estimation, dimension))
        mean, _ = self._predict(surrogate, estimation)
        failures = int(np.sum(mean <= 0))
        probability = failures / self.n_estimation
        cov = (
            np.sqrt((1 - probability) / (self.n_estimation * probability))
            if 0 < failures < self.n_estimation
            else np.inf
        )
        interval = binomtest(failures, self.n_estimation).proportion_ci(
            confidence_level=0.95
        )
        converged = bool(learned and cov <= self.target_cov)
        if learned:
            status = "converged" if converged else "sampling_precision"
        self.result = ActiveLearningResult(
            probability,
            float(-norm.ppf(probability)),
            float(cov),
            (float(interval.low), float(interval.high)),
            converged,
            status,
            len(design),
            self.n_estimation,
            tuple(history),
        )
        self.surrogate_model = surrogate
        self.results_valid = converged
        if not converged:
            warnings.warn(
                f"Active learning did not converge: {status}",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.result
