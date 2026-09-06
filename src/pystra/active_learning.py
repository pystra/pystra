"""Optional active-learning reliability in independent normal coordinates.

AK-MCS: Echard, Gayton and Lemaire (2011), doi:10.1016/j.strusafe.2011.01.002.
EFF: Bichon et al. (2008), doi:10.2514/1.34321.
Bootstrap PCE: inspired by Marelli and Sudret (2018),
doi:10.1016/j.strusafe.2018.06.003. Here the total-degree basis is fixed,
without their sparse selection, degree adaptation or batch enrichment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb, factorial
from typing import Optional, Union
import warnings

import numpy as np
from scipy.special import eval_hermitenorm
from scipy.stats import norm, qmc

from .analysis import AnalysisObject

__all__ = [
    "ActiveLearning",
    "ActiveLearningResult",
    "LearningStep",
    "Surrogate",
    "KrigingSurrogate",
    "PceSurrogate",
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


class PceSurrogate(Surrogate):
    """Fixed total-degree Hermite PCE with pairs-bootstrap local spread.

    ``degree`` defaults to 3; ``n_bootstrap`` defaults to 30 (at least 2).
    The mean uses the full design least-squares fit. Standard deviation
    uses bootstrap predictions with ddof=1. ``seed`` controls resampling.
    Fits require an overdetermined, full-rank design. Rank-deficient
    bootstrap draws are retried, then fail explicitly if insufficient.

    This is a dense, fixed-basis variant inspired by Marelli and Sudret
    (2018), not a reproduction of their adaptive sparse-PCE algorithm.
    Bootstrap spread does not detect shared polynomial truncation bias
    and is not a Gaussian posterior or a guaranteed error bound.
    """

    def __init__(
        self, *, degree: int = 3, n_bootstrap: int = 30, seed: Optional[int] = None
    ):
        self.degree = _positive_integer(degree, "degree")
        self.n_bootstrap = _positive_integer(n_bootstrap, "n_bootstrap", 2)
        self.seed = seed
        self._coefficients = None

    def _basis(self, points):
        basis = np.ones((len(points), len(self._powers)))
        for column, powers in enumerate(self._powers):
            for axis, power in enumerate(powers):
                if power:
                    basis[:, column] *= eval_hermitenorm(
                        power, points[:, axis]
                    ) / np.sqrt(float(factorial(power)))
        return basis

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        """Refit the basis and bootstrap ensemble in normal coordinates."""
        self._coefficients = None
        points, values = _training(points, values)
        self._dimension = points.shape[1]
        self._powers = []
        for degree in range(self.degree + 1):
            for indices in combinations_with_replacement(
                range(self._dimension), degree
            ):
                self._powers.append(np.bincount(indices, minlength=self._dimension))
        basis = self._basis(points)
        if len(points) <= basis.shape[1]:
            raise ValueError(
                "PCE requires more training points than basis terms; increase n_initial or reduce degree"
            )
        coefficients, _, rank, _ = np.linalg.lstsq(basis, values, rcond=None)
        if rank != basis.shape[1]:
            raise ValueError("PCE training design is rank deficient")
        rng = np.random.default_rng(self.seed)
        ensemble = []
        for _ in range(100 * self.n_bootstrap):
            indices = rng.integers(len(points), size=len(points))
            fitted, _, rank, _ = np.linalg.lstsq(
                basis[indices], values[indices], rcond=None
            )
            if rank == basis.shape[1]:
                ensemble.append(fitted)
            if len(ensemble) == self.n_bootstrap:
                break
        if len(ensemble) != self.n_bootstrap:
            raise ValueError(
                "Insufficient full-rank bootstrap designs; increase n_initial"
            )
        self._ensemble = np.asarray(ensemble).T
        self._coefficients = coefficients

    def predict(self, points: np.ndarray) -> tuple:
        """Return full-design mean and local bootstrap prediction spread."""
        if self._coefficients is None:
            raise RuntimeError("Surrogate has not been fitted")
        points = _points(points)
        if points.shape[1] != self._dimension:
            raise ValueError("Prediction dimension differs from training dimension")
        basis = self._basis(points)
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
        Initial LHS size: max(12, 2*n_variables), or at least twice the PCE
        basis size. PCE degree is specified in surrogate_kwargs.
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
            initial = max(
                initial, 2 * comb(dimension + surrogate.degree, surrogate.degree)
            )
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
