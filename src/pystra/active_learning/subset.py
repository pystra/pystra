"""Replicated subset simulation and estimator-driven active learning.

The nested-event construction follows Au and Beck (2001),
doi:10.1016/S0266-8920(01)00019-4. Gaussian-preserving pCN proposals replace
componentwise modified Metropolis (Cotter et al., 2013,
doi:10.1214/13-STS421). Active enrichment follows the modular
framework of Moustapha, Marelli and Sudret (2022),
doi:10.1016/j.strusafe.2021.102174; this is a documented variant of that workflow.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ._validation import _positive_integer, _predictions
from .estimation import EnrichmentEstimator, EnrichmentResult
from .results import ReliabilityEstimate


@dataclass(frozen=True)
class SubsetLevel:
    """One level's threshold, conditional probability and chain diagnostics.

    The final probability concerns mean <= 0, while exploratory intermediate
    levels condition on mean - 2 std. ``variance_factor`` compares a chain
    cluster variance estimate with IID Bernoulli variance, floored at one.
    It does not account for ancestry shared by chains at different levels.
    ``acceptance_rate`` concerns proposals generating this level (one at the
    initial IID level). Seeds are included in each conditional chain.
    """

    threshold: float
    conditional_probability: float
    n_samples: int
    n_chains: int
    variance_factor: float
    acceptance_rate: float


@dataclass(frozen=True)
class SubsetRun:
    """One independent replication, including incomplete-run diagnostics.

    ``sampling_cov`` is a within-chain approximation with an IID floor; it
    omits cross-level dependence. The aggregate estimator also uses variation
    between independent complete runs. Probability bands are computed using
    the same nested measure, not pooled conditional sample proportions.
    """

    failure_probability: float
    probability_band: tuple
    sampling_cov: float
    n_evaluations: int
    levels: tuple
    converged: bool
    status: str


def _indicator_variance(indicators, chains):
    """Variance of a sample proportion allowing within-chain dependence.

    Treat chain sums as clusters, with a finite-cluster correction. Independent
    seeds/levels are only an approximation; independent full replications are
    used separately to detect the omitted dependence. Do not report reduced
    variance from noisy negative correlation estimates.
    """
    indicators = np.asarray(indicators, dtype=float)
    probability = float(np.mean(indicators))
    iid = probability * (1 - probability) / len(indicators)
    if chains is None or iid == 0:
        return iid, 1.0
    n_chains = int(chains.max()) + 1
    if n_chains < 2:
        return np.inf, np.inf
    residuals = np.bincount(chains, weights=indicators - probability)
    variance = (
        float(residuals @ residuals) * n_chains / (n_chains - 1) / len(indicators) ** 2
    )
    return max(iid, variance), max(1.0, variance / iid)


def _guide(mean, std, spread):
    try:
        with np.errstate(over="raise", invalid="raise"):
            return mean - spread * std
    except FloatingPointError as exc:
        raise ValueError("Surrogate prediction band overflowed") from exc


class SubsetSimulationEstimator(EnrichmentEstimator):
    """Subset simulation with replicated sampling and adaptive enrichment.

    Parameters
    ----------
    n_samples : int
        Samples per subset level per replication (default 2000, minimum 20).
        Used during exploration and independent final estimation.
    conditional_probability : float
        Target intermediate probability, default 0.1, in (0, 1).
        At least two seeds and two nonseeds must result at this sample size.
    proposal_scale : float
        pCN innovation standard deviation in (0, 1], default 0.5. The proposal
        is sqrt(1-scale**2)*u + scale*z, z ~ N(0,I). It is reversible for the
        normal prior; only the conditioning-event acceptance test is needed.
    max_levels : int
        Maximum levels per replication, including the IID level (default 12).
        Unreached failure or stalled positive thresholds are explicit failures.
    n_replications : int
        Independent complete subset runs (default 4, minimum 2). Their mean
        supplies Pf. Final CoV uses the larger of the empirical standard error
        across complete runs and the aggregated within-chain approximation.

    Notes
    -----
    Inputs are row-wise independent standard normal coordinates; failure is
    mean <= 0. Exploration conditions on mean - 2 std, so mean +/- 2 std
    failure events are nested inside every positive intermediate event. All
    three probabilities use the same prefix probability and final conditional
    population. Pooled levels are used only to select enrichment points.

    Exploration restarts after every fit with common random numbers. Final
    estimation starts independent replications on the frozen surrogate mean.
    The CoV is an estimate, not a bound: a small replication count, shared
    surrogate bias, and missing disconnected modes can still mislead. No
    binomial confidence interval is assigned to dependent subset samples.
    """

    def __init__(
        self,
        *,
        n_samples: int = 2000,
        conditional_probability: float = 0.1,
        proposal_scale: float = 0.5,
        max_levels: int = 12,
        n_replications: int = 4,
    ):
        self.n_samples = _positive_integer(n_samples, "n_samples", 20)
        if (
            not np.isfinite(conditional_probability)
            or not 0 < conditional_probability < 1
        ):
            raise ValueError("conditional_probability must be in (0, 1)")
        self.conditional_probability = float(conditional_probability)
        self._n_seeds = int(np.ceil(self.n_samples * conditional_probability))
        if not 2 <= self._n_seeds <= self.n_samples - 2:
            raise ValueError(
                "conditional_probability must give at least two seeds and nonseeds"
            )
        if not np.isfinite(proposal_scale) or not 0 < proposal_scale <= 1:
            raise ValueError("proposal_scale must be in (0, 1]")
        self.proposal_scale = float(proposal_scale)
        self.max_levels = _positive_integer(max_levels, "max_levels")
        self.n_replications = _positive_integer(n_replications, "n_replications", 2)

    def _predict(self, predict, points):
        mean, std = _predictions(*predict(points))
        if len(mean) != len(points):
            raise ValueError("Predictor returned the wrong number of predictions")
        return mean, std

    def _conditional(self, predict, points, mean, std, threshold, spread, rng):
        order = rng.permutation(len(points))
        points, mean, std = points[order], mean[order], std[order]
        n_chains, dimension = points.shape
        lengths = np.full(n_chains, self.n_samples // n_chains)
        lengths[: self.n_samples % n_chains] += 1
        width = int(lengths.max())
        states = np.empty((n_chains, width, dimension))
        means = np.empty((n_chains, width))
        spreads = np.empty((n_chains, width))
        states[:, 0], means[:, 0], spreads[:, 0] = points, mean, std
        accepted = attempts = 0
        persistence = np.sqrt(1 - self.proposal_scale**2)
        for position in range(1, width):
            active = np.flatnonzero(lengths > position)
            proposal = persistence * points[
                active
            ] + self.proposal_scale * rng.standard_normal((len(active), dimension))
            proposed_mean, proposed_std = self._predict(predict, proposal)
            inside = _guide(proposed_mean, proposed_std, spread) <= threshold
            selected = active[inside]
            points[selected] = proposal[inside]
            mean[selected], std[selected] = proposed_mean[inside], proposed_std[inside]
            states[active, position] = points[active]
            means[active, position], spreads[active, position] = (
                mean[active],
                std[active],
            )
            accepted += int(inside.sum())
            attempts += len(active)
        mask = np.arange(width)[None, :] < lengths[:, None]
        return (
            states[mask],
            means[mask],
            spreads[mask],
            np.repeat(np.arange(n_chains), lengths),
            accepted / attempts,
            attempts,
        )

    def _run(self, predict, dimension, rng, *, spread, collect):
        points = rng.standard_normal((self.n_samples, dimension))
        mean, std = self._predict(predict, points)
        evaluations = self.n_samples
        chains = None
        acceptance = 1.0
        prefix = 1.0
        previous = np.inf
        variance_terms = []
        levels, population = [], []
        for level in range(self.max_levels):
            if collect:
                population.append((points.copy(), mean.copy(), std.copy()))
            guide = _guide(mean, std, spread)
            threshold = float(np.partition(guide, self._n_seeds - 1)[self._n_seeds - 1])
            complete = threshold <= 0
            stalled = not complete and (
                threshold >= previous or np.all(guide <= threshold)
            )
            last = complete or stalled or level == self.max_levels - 1
            indicators = mean <= 0 if last else guide <= threshold
            probability = float(np.mean(indicators))
            variance, factor = _indicator_variance(indicators, chains)
            variance_terms.append(
                variance / probability**2 if 0 < probability < 1 else np.inf
            )
            levels.append(
                SubsetLevel(
                    threshold=0.0 if complete else threshold,
                    conditional_probability=probability,
                    n_samples=self.n_samples,
                    n_chains=(
                        self.n_samples if chains is None else int(chains.max()) + 1
                    ),
                    variance_factor=float(factor),
                    acceptance_rate=float(acceptance),
                )
            )
            if last:
                lower = float(prefix * np.mean(_guide(mean, std, -spread) <= 0))
                upper = float(prefix * np.mean(guide <= 0))
                run = SubsetRun(
                    failure_probability=float(prefix * probability),
                    probability_band=(lower, upper),
                    sampling_cov=float(np.sqrt(sum(variance_terms))),
                    n_evaluations=evaluations,
                    levels=tuple(levels),
                    converged=bool(complete),
                    status=(
                        "completed"
                        if complete
                        else ("stalled" if stalled else "max_levels")
                    ),
                )
                return run, population
            prefix *= probability
            previous = threshold
            selected = guide <= threshold
            points, mean, std, chains, acceptance, count = self._conditional(
                predict,
                points[selected],
                mean[selected],
                std[selected],
                threshold,
                spread,
                rng,
            )
            evaluations += count

    def _replicate(self, predict, dimension, rng, *, spread, collect):
        dimension = _positive_integer(dimension, "dimension")
        runs, population = [], []
        for seed in rng.integers(0, 2**63 - 1, size=self.n_replications):
            run, samples = self._run(
                predict,
                dimension,
                np.random.default_rng(seed),
                spread=spread,
                collect=collect,
            )
            runs.append(run)
            population.extend(samples)
        probabilities = np.array([run.failure_probability for run in runs])
        lower, probability, upper = (
            float(value)
            for value in np.mean(
                [
                    (
                        run.probability_band[0],
                        run.failure_probability,
                        run.probability_band[1],
                    )
                    for run in runs
                ],
                axis=0,
            )
        )
        completed = all(run.converged for run in runs)
        cov = np.inf
        if completed and 0 < probability < 1 and np.all(probabilities > 0):
            between = float(np.std(probabilities, ddof=1) / np.sqrt(len(runs)))
            within = float(
                np.sqrt(
                    sum(
                        (run.failure_probability * run.sampling_cov) ** 2
                        for run in runs
                    )
                )
                / len(runs)
            )
            cov = max(between, within) / probability
        estimate = ReliabilityEstimate(
            failure_probability=probability,
            sampling_cov=float(cov),
            sampling_interval=None,
            confidence_level=None,
            n_samples=sum(run.n_evaluations for run in runs),
            method="subset_simulation",
            sampling_dependence=(
                "dependent"
                if any(len(run.levels) > 1 for run in runs)
                else "independent"
            ),
            converged=completed,
            status=(
                "completed"
                if completed
                else next(run.status for run in runs if not run.converged)
            ),
            diagnostics=tuple(runs),
        )
        return estimate, (lower, upper), population

    def estimate(
        self,
        predict: Callable[[np.ndarray], tuple],
        *,
        dimension: int,
        rng: np.random.Generator,
    ) -> ReliabilityEstimate:
        """Run independent replications on the frozen mean failure event."""
        return self._replicate(predict, dimension, rng, spread=0.0, collect=False)[0]

    def explore(
        self,
        predict: Callable[[np.ndarray], tuple],
        *,
        dimension: int,
        rng: np.random.Generator,
        n_candidates: int,
    ) -> EnrichmentResult:
        """Resample optimistic subsets and pool their states for enrichment.

        Retain at most n_candidates unique points sampled uniformly from the
        union of level states. The pool cap changes selection cost only, not
        the probability calculation. Reject duplicate true observations in
        the runner, after sampling and before selection.
        """
        n_candidates = _positive_integer(n_candidates, "n_candidates")
        estimate, band, population = self._replicate(
            predict, dimension, rng, spread=2.0, collect=True
        )
        points = np.vstack([samples[0] for samples in population])
        mean = np.concatenate([samples[1] for samples in population])
        std = np.concatenate([samples[2] for samples in population])
        _, indices = np.unique(points, axis=0, return_index=True)
        if len(indices) > n_candidates:
            indices = rng.choice(indices, size=n_candidates, replace=False)
        return EnrichmentResult(
            points[indices], mean[indices], std[indices], estimate, band
        )
