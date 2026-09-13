"""Compose active-learning reliability components."""

from math import comb
import warnings

import numpy as np
from scipy.stats import norm, qmc

from ..reliability.analysis import AnalysisObject, _check_rng, _generator
from ..options import SimulationOptions
from ..model import LimitState, StochasticModel
from ._validation import _positive_integer, _points, _training, _predictions
from .surrogates import Surrogate, KrigingSurrogate, PCESurrogate, EnsembleSurrogate
from .pc_kriging import PCKrigingSurrogate
from .learning import (
    LearningDecision,
    LearningFunction,
    UFunction,
    ExpectedFeasibility,
    EnsembleLearningFunction,
    FBRLearning,
    _replicates,
)
from .estimation import (
    ReliabilityEstimator,
    MonteCarloEstimator,
    EnrichmentEstimator,
    EnrichmentResult,
)
from .stopping import StoppingCriterion, LearningThreshold, BootstrapBounds
from .results import ActiveLearningResult, LearningStep, ReliabilityEstimate

__all__ = ["ActiveLearning"]


class ActiveLearning(AnalysisObject):
    """Enrich a surrogate in independent normal coordinates, then estimate Pf.

    Parameters
    ----------
    model : StochasticModel
    limit_state : LimitState
        Standard PySTRA analysis inputs. Failure is g <= 0.
    options : SimulationOptions, optional
        Block size and transformation. Sample sizes and precision are this
        class's own arguments, so ``n_samples``, ``target_cov``,
        ``sampling_std`` and ``bins`` are rejected.
    surrogate : str or Surrogate
        'kriging' (default), 'pce', 'pc_kriging', or an explicitly supplied fitted-model
        implementation. A supplied object is refitted in place by run().
    learning_function : str or LearningFunction
        'u' (default), 'eff', 'fbr', or a stateless selection policy.
        EFF assumes Gaussian predictive uncertainty. FBR requires bootstrap
        replicates; its default stopping policy is BootstrapBounds.
    n_initial : int, optional
        Initial LHS size: max(12, 2*n_variables), or max(30, 5*n_variables)
        for sparse PCE or estimator-driven enrichment. Dense OLS uses at least twice the largest total-degree
        basis size.
    n_candidates : int
        Fixed MC pool size, or maximum size of each estimator-generated
        enrichment pool (10000). Subset sample size belongs to the estimator.
    n_estimation : int, optional
        Shortcut for the default MonteCarloEstimator population (100000).
        Cannot be combined with an explicit estimator.
    estimator : ReliabilityEstimator, optional
        Default MonteCarloEstimator(). Owns its sampling uncertainty.
        EnrichmentEstimator implementations also resample the enrichment
        pool after every fit and supply measure-correct probability bands.
    max_iterations : int
        Maximum added true evaluations (200), excluding the initial design.
    learning_threshold : float, optional
        U minimum (2), or EFF maximum (1e-3, in limit-state units).
        Only valid for a named learning function.
    target_cov : float, optional
        Shortcut for the default LearningThreshold's final sampling CoV (0.1).
        Cannot be combined with an explicit stopping criterion.
    stopping_criterion : StoppingCriterion, optional
        Default LearningThreshold(), or BootstrapBounds() for FBR.
        Explicit policies separate convergence
        from selection, for example AllCriteria with beta bounds/stability.
    surrogate_kwargs : mapping, optional
        Constructor settings for a named surrogate.
    rng : int, numpy.random.Generator or None, optional
        Random source of the run-owned LHS, candidates, final population and
        named surrogate optimizer. A seed repeats them on every run; a
        generator advances its own state. Adaptive exploration reuses a
        separate seed after each fit; final estimation is independent of that
        stream.

    Notes
    -----
    Candidate points are evaluated at most once. Predictions are batched.
    Nonconvergence warns and returns an explicit unfinished result. Exceptions
    invalidate any prior result. A finite pool may miss disconnected failures;
    bootstrap spread can miss common polynomial bias. Validate the surrogate
    independently for the problem being assessed.
    """

    _options_type = SimulationOptions

    def __init__(
        self,
        model: StochasticModel,
        limit_state: LimitState,
        *,
        options: SimulationOptions | None = None,
        surrogate: str | Surrogate = "kriging",
        learning_function: str | LearningFunction = "u",
        n_initial: int | None = None,
        n_candidates: int = 10_000,
        n_estimation: int | None = None,
        estimator: ReliabilityEstimator | None = None,
        max_iterations: int = 200,
        learning_threshold: float | None = None,
        target_cov: float | None = None,
        stopping_criterion: StoppingCriterion | None = None,
        surrogate_kwargs: dict | None = None,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(model, limit_state, options)
        self.options._require_defaults(
            "ActiveLearning", ("n_samples", "target_cov", "sampling_std", "bins")
        )
        if not isinstance(surrogate, Surrogate) and surrogate not in (
            "kriging",
            "pce",
            "pc_kriging",
        ):
            raise ValueError(
                "Unknown surrogate; use 'kriging', 'pce', 'pc_kriging', or a Surrogate"
            )
        if isinstance(learning_function, LearningFunction):
            if learning_threshold is not None:
                raise ValueError(
                    "learning_threshold requires a named learning function"
                )
        elif learning_function == "fbr":
            if learning_threshold is not None:
                raise ValueError(
                    "FBR uses bootstrap probability stopping, not learning_threshold"
                )
            learning_function = FBRLearning()
        elif isinstance(learning_function, str) and learning_function in ("u", "eff"):
            policy = UFunction if learning_function == "u" else ExpectedFeasibility
            learning_function = (
                policy()
                if learning_threshold is None
                else policy(threshold=learning_threshold)
            )
        else:
            raise ValueError("Use 'u', 'eff', 'fbr', or a LearningFunction")
        if estimator is None:
            estimator = MonteCarloEstimator(
                n_samples=100_000 if n_estimation is None else n_estimation
            )
        elif not isinstance(estimator, ReliabilityEstimator):
            raise ValueError("estimator must be a ReliabilityEstimator")
        elif n_estimation is not None:
            raise ValueError(
                "n_estimation cannot be combined with an explicit estimator"
            )
        if stopping_criterion is None:
            policy = (
                BootstrapBounds
                if isinstance(learning_function, FBRLearning)
                else LearningThreshold
            )
            stopping_criterion = policy(
                target_cov=0.1 if target_cov is None else target_cov
            )
        elif not isinstance(stopping_criterion, StoppingCriterion):
            raise ValueError("stopping_criterion must be a StoppingCriterion")
        elif target_cov is not None:
            raise ValueError(
                "target_cov cannot be combined with an explicit stopping criterion"
            )
        if isinstance(surrogate, Surrogate) and surrogate_kwargs:
            raise ValueError("surrogate_kwargs requires a named surrogate")
        needs_replicates = (
            isinstance(learning_function, EnsembleLearningFunction)
            or stopping_criterion.requires_bootstrap
        )
        if needs_replicates and not (
            isinstance(surrogate, EnsembleSurrogate) or surrogate == "pce"
        ):
            raise TypeError("Bootstrap learning/stopping requires an EnsembleSurrogate")
        if stopping_criterion.requires_bootstrap and isinstance(
            estimator, EnrichmentEstimator
        ):
            raise ValueError("BootstrapBounds requires fixed IID normal enrichment")
        self.surrogate = surrogate
        self.learning_function = learning_function
        self.estimator = estimator
        self.stopping_criterion = stopping_criterion
        self.n_initial = (
            None if n_initial is None else _positive_integer(n_initial, "n_initial", 2)
        )
        self.n_candidates = _positive_integer(n_candidates, "n_candidates")
        self.max_iterations = _positive_integer(max_iterations, "max_iterations", 0)
        self.surrogate_kwargs = dict(surrogate_kwargs or {})
        self.rng = _check_rng(rng)
        self.result = None
        self.surrogate_model = None

    def _evaluate(self, points):
        marginals = self.model.marginal_distributions
        physical = np.asarray(
            [self.transform.u_to_x(point, marginals) for point in points]
        )
        values, _ = self._lsf(physical.T)
        return _training(points, np.asarray(values).ravel())[1]

    def _predict(self, surrogate, points):
        points = _points(points)
        if points.shape[1] != self.model.n_marg:
            raise ValueError("Prediction dimension differs from the stochastic model")
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
        self.result = None
        self.surrogate_model = None
        self._results_valid = False
        self.init_run()
        dimension = self.model.n_marg
        if dimension < 1:
            raise ValueError("Active learning requires at least one random variable")
        rng = _generator(self.rng)
        settings = dict(self.surrogate_kwargs)
        settings.setdefault("seed", int(rng.integers(2**31 - 1)))
        surrogate = self.surrogate
        if isinstance(surrogate, str):
            surrogate = {
                "kriging": KrigingSurrogate,
                "pce": PCESurrogate,
                "pc_kriging": PCKrigingSurrogate,
            }[surrogate](**settings)
        initial = max(12, 2 * dimension)
        if isinstance(self.estimator, EnrichmentEstimator):
            initial = max(initial, 30, 5 * dimension)
        if isinstance(surrogate, PCKrigingSurrogate):
            initial = max(30, 5 * dimension)
        if isinstance(surrogate, PCESurrogate):
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
        adaptive = isinstance(self.estimator, EnrichmentEstimator)
        if adaptive:
            exploration_seed = int(rng.integers(0, 2**63 - 1))
        else:
            candidates = rng.standard_normal((self.n_candidates, dimension))
            available = np.ones(self.n_candidates, dtype=bool)
        history = []
        learned = False
        status = "max_iterations"
        for iteration in range(self.max_iterations + 1):
            surrogate.fit(design, observations)
            exploration = None
            if adaptive:
                exploration = self.estimator.explore(
                    lambda points: self._predict(surrogate, points),
                    dimension=dimension,
                    rng=np.random.default_rng(exploration_seed),
                    n_candidates=self.n_candidates,
                )
                if not isinstance(exploration, EnrichmentResult):
                    raise TypeError(
                        "EnrichmentEstimator must return an EnrichmentResult"
                    )
                candidates, mean, std = (
                    exploration.points,
                    exploration.mean,
                    exploration.std,
                )
                if candidates.shape[1] != dimension:
                    raise ValueError(
                        "Enrichment dimension differs from the stochastic model"
                    )
                observed = {tuple(point) for point in design}
                available = np.array(
                    [tuple(point) not in observed for point in candidates]
                )
            else:
                mean, std = self._predict(surrogate, candidates)
            indices = np.flatnonzero(available)
            if not len(indices):
                status = "candidate_exhaustion"
                break
            bootstrap_band = None
            replicates = None
            if (
                isinstance(self.learning_function, EnsembleLearningFunction)
                or self.stopping_criterion.requires_bootstrap
            ):
                batches = []
                for start in range(0, len(candidates), 2048):
                    batch = candidates[start : start + 2048]
                    predictions = _replicates(surrogate.predict_replicates(batch))
                    if len(predictions) != len(batch):
                        raise ValueError(
                            "Surrogate returned the wrong number of replicate predictions"
                        )
                    batches.append(predictions)
                replicates = np.concatenate(batches)
                if exploration is None:
                    probabilities = np.mean(replicates <= 0, axis=0)
                    bootstrap_band = (
                        float(probabilities.min()),
                        float(probabilities.max()),
                    )
            if isinstance(self.learning_function, EnsembleLearningFunction):
                decision = self.learning_function.select_replicates(replicates[indices])
            else:
                decision = self.learning_function.select(mean[indices], std[indices])
            if not isinstance(decision, LearningDecision):
                raise TypeError("LearningFunction must return a LearningDecision")
            if decision.index >= len(indices):
                raise ValueError(
                    "LearningFunction selected an index outside the available pool"
                )
            if exploration is None:
                # Only the fixed IID normal pool admits unweighted proportions.
                with np.errstate(over="ignore"):
                    lower = float(np.mean(mean + 2 * std <= 0))
                    upper = float(np.mean(mean - 2 * std <= 0))
                probability = float(np.mean(mean <= 0))
                estimation_converged, sampling_cov = True, None
            else:
                lower, upper = exploration.probability_band
                probability = exploration.estimate.failure_probability
                estimation_converged = exploration.estimate.converged
                sampling_cov = exploration.estimate.sampling_cov
            history.append(
                LearningStep(
                    failure_probability=probability,
                    learning_score=decision.score,
                    n_limit_state_evaluations=len(design),
                    probability_band=(lower, upper),
                    beta_band=(float(-norm.ppf(upper)), float(-norm.ppf(lower))),
                    learning_satisfied=decision.threshold_satisfied,
                    estimation_converged=estimation_converged,
                    sampling_cov=sampling_cov,
                    bootstrap_probability_band=bootstrap_band,
                )
            )
            learned = bool(
                estimation_converged
                and self.stopping_criterion.should_stop(tuple(history))
            )
            if learned or iteration == self.max_iterations:
                break
            selected = indices[decision.index]
            point = candidates[selected : selected + 1]
            observations = np.concatenate((observations, self._evaluate(point)))
            design = np.vstack((design, point))
            available[np.all(candidates == point, axis=1)] = False
        # The estimator owns final sampling and its uncertainty calculation.
        estimate = self.estimator.estimate(
            lambda points: self._predict(surrogate, points),
            dimension=dimension,
            rng=rng,
        )
        if not isinstance(estimate, ReliabilityEstimate):
            raise TypeError("ReliabilityEstimator must return a ReliabilityEstimate")
        converged = bool(
            learned
            and estimate.converged
            and self.stopping_criterion.accepts_estimate(estimate)
        )
        if learned:
            status = (
                "converged"
                if converged
                else (
                    "sampling_precision" if estimate.converged else "estimation_failed"
                )
            )
        self.result = ActiveLearningResult(
            estimate=estimate,
            converged=converged,
            status=status,
            n_limit_state_evaluations=len(design),
            history=tuple(history),
        )
        self.surrogate_model = surrogate
        self._results_valid = converged
        if not converged:
            warnings.warn(
                f"Active learning did not converge: {status}",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.result
