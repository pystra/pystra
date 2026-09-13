# -*- coding: utf-8 -*-
"""Subset Simulation reliability analysis."""

import numpy as np
from scipy.stats import norm as scipy_norm

from .analysis import AnalysisObject, _check_rng, _generator
from ..options import SimulationOptions
from ..results import SimulationResult

__all__ = ["SubsetSimulation"]


class SubsetSimulation(AnalysisObject):
    r"""Subset Simulation (SS) reliability analysis.

    Subset Simulation decomposes the rare failure event
    :math:`F = \{g(\mathbf{u}) \le 0\}` into a sequence of more frequent
    nested intermediate events :math:`F_1 \supset F_2 \supset \cdots
    \supset F_m = F`:

    .. math::

       p_f = P(F_1)\prod_{j=2}^{m} P(F_j \mid F_{j-1})

    The intermediate thresholds :math:`y_1 > y_2 > \cdots > y_m = 0` are
    chosen adaptively so that each conditional probability is approximately
    equal to the target level :math:`p_0`.  Conditional samples are generated
    via the Modified Metropolis–Hastings (MMH) algorithm operating
    component-wise in standard normal space.

    Parameters
    ----------
    model : StochasticModel
    limit_state : LimitState
    options : SimulationOptions, optional
        ``n_samples`` is the number of samples per level.
    p0 : float, optional
        Target conditional failure probability per subset level (default 0.1).
    proposal_sigma : float, optional
        Half-width of the uniform proposal kernel used in MMH, measured in
        standard-deviation units of the standard normal space (default 1.0).
    rng : int, numpy.random.Generator or None, optional
        Random source; NumPy's global generator is not used. A seed recreates
        the same stream on every run, a generator advances its own state, and
        None draws fresh entropy.

    Notes
    -----
    The number of samples per level is ``options.n_samples``. :meth:`run`
    returns a :class:`~pystra.results.SimulationResult`; its diagnostics
    include the thresholds and conditional probabilities.

    References
    ----------
    Au, S. K., & Beck, J. L. (2001).  Estimation of small failure
    probabilities in high dimensions by subset simulation.
    *Probabilistic Engineering Mechanics*, 16(4), 263–277.
    """

    _options_type = SimulationOptions

    def __init__(
        self, model, limit_state, *, options=None, p0=0.1, proposal_sigma=1.0, rng=None
    ):
        super().__init__(model, limit_state, options)
        self.rng = _check_rng(rng)
        self.options._require_defaults(
            "SubsetSimulation", ("target_cov", "sampling_std", "bins")
        )
        if not (0.0 < p0 < 1.0):
            raise ValueError(f"p0 must be in (0, 1); got {p0}")
        self.p0 = p0
        self.proposal_sigma = proposal_sigma
        self._Pf = None
        self._beta = None
        self._cov = None
        self._thresholds = []
        self._conditional_probs = []
        self._n_levels = None

    def run(self):
        """Run subset simulation and return a :class:`SimulationResult`."""
        self._results_valid = True
        self.init_run()
        self._random = _generator(self.rng)

        nrv = self.model.get_len_marginal_distributions()
        marg = self.model.get_marginal_distributions()
        N = self.options.n_samples
        p0 = self.p0

        self._thresholds = []
        self._conditional_probs = []

        # ---------------------------------------------------------------
        # Level 0: direct Monte Carlo from the prior N(0, I)
        # ---------------------------------------------------------------
        u = self._random.standard_normal((nrv, N))
        G = self._eval_g_batch(u, marg)

        # y_1 = p0-th quantile of G (the p0 fraction closest to failure G<0)
        y = float(np.percentile(G, p0 * 100.0))

        if y <= 0.0:
            # Threshold already at or below failure: level-0 MC suffices
            Pf = float(np.sum(G <= 0.0)) / N
            self._thresholds.append(0.0)
            self._conditional_probs.append(Pf)
            self._n_levels = 1
            return self._finalise(Pf, N)

        self._thresholds.append(y)
        p_lvl = float(np.sum(G <= y)) / N
        self._conditional_probs.append(p_lvl)
        Pf = p_lvl

        # Seeds: samples that satisfy the first intermediate event G <= y_1
        seed_mask = G <= y
        seeds_u = u[:, seed_mask]
        seeds_G = G[seed_mask]

        # ---------------------------------------------------------------
        # Successive subset levels
        # ---------------------------------------------------------------
        level = 1
        while True:
            # Generate N conditional samples via MMH from current seeds
            u_new, G_new = self._mmh_step(seeds_u, seeds_G, y, N, nrv, marg)
            y_new = float(np.percentile(G_new, p0 * 100.0))

            if y_new <= 0.0:
                # Last level: count actual failures (G <= 0)
                p_last = float(np.sum(G_new <= 0.0)) / N
                Pf = Pf * p_last
                self._thresholds.append(0.0)
                self._conditional_probs.append(p_last)
                break

            # Intermediate level
            p_lvl = float(np.sum(G_new <= y_new)) / N
            Pf = Pf * p_lvl
            self._thresholds.append(y_new)
            self._conditional_probs.append(p_lvl)

            # New seeds for the next level
            seed_mask = G_new <= y_new
            seeds_u = u_new[:, seed_mask]
            seeds_G = G_new[seed_mask]

            y = y_new
            level += 1

        self._n_levels = level + 1
        return self._finalise(Pf, N)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finalise(self, Pf, N):
        """Store results, estimate the CoV and return the record."""
        self._Pf = float(Pf)
        if 0.0 < Pf < 1.0:
            self._beta = float(-scipy_norm.ppf(Pf))
        elif Pf <= 0.0:
            self._beta = np.inf
        else:
            self._beta = -np.inf

        # CoV lower bound (γ_j = 0, i.e. ignoring Markov-chain correlations)
        # δ²(Pf) ≈ Σ_j (1 - p_j) / (N * p_j)
        delta_sq = sum((1.0 - p) / (N * p) for p in self._conditional_probs if p > 0.0)
        self._cov = float(np.sqrt(delta_sq)) if delta_sq > 0.0 else 0.0

        return SimulationResult(
            method="SubsetSimulation",
            status="completed",
            message="Sampling completed",
            n_limit_state_evaluations=self._n_evaluations,
            variable_names=tuple(self.model.get_variables()),
            failure_probability=self._Pf,
            beta=self._beta,
            coefficient_of_variation=self._cov if self._Pf > 0 else np.inf,
            n_samples=N * self._n_levels,
            options=self.options,
            diagnostics={
                "thresholds": np.array(self._thresholds),
                "conditional_probabilities": np.array(self._conditional_probs),
                "n_levels": self._n_levels,
                "samples_per_level": N,
            },
        )

    def _eval_g_batch(self, u, marg):
        """Evaluate the LSF for every column of *u* (shape nrv × N)."""
        N = u.shape[1]
        G = np.empty(N)
        block = self.options.block_size
        for start in range(0, N, block):
            end = min(start + block, N)
            u_blk = u[:, start:end]
            x_blk = np.empty_like(u_blk)
            for i in range(end - start):
                x_blk[:, i] = self.transform.u_to_x(u_blk[:, i], marg)
            G_blk, _ = self._lsf(x_blk)
            G[start:end] = G_blk[0, :]
        return G

    def _mmh_step(self, seeds, seeds_G, threshold, N, nrv, marg):
        """Generate *N* samples from N(0, I) conditioned on g ≤ *threshold*.

        Uses the Modified Metropolis–Hastings algorithm with a uniform
        proposal kernel of half-width ``proposal_sigma`` applied
        component-wise.

        Parameters
        ----------
        seeds : ndarray, shape (nrv, N_seeds)
            Seed samples (all satisfying g ≤ threshold).
        seeds_G : ndarray, shape (N_seeds,)
            LSF values at the seeds.
        threshold : float
            Current intermediate threshold.
        N : int
            Number of output samples desired.
        nrv : int
        marg : list
        """
        N_seeds = seeds.shape[1]
        sigma = self.proposal_sigma

        u_out = np.empty((nrv, N))
        G_out = np.empty(N)

        # Distribute N samples across N_seeds chains as evenly as possible
        base = N // N_seeds
        remainder = N - base * N_seeds

        idx = 0
        for s in range(N_seeds):
            n_chain = base + (1 if s < remainder else 0)
            u_curr = seeds[:, s].copy()
            g_curr = seeds_G[s]

            for _ in range(n_chain):
                # Component-wise Metropolis step with uniform proposal
                u_prop = u_curr.copy()
                for d in range(nrv):
                    xi = u_curr[d] + sigma * self._random.uniform(-1.0, 1.0)
                    # Accept with min(1, phi(xi)/phi(u_curr[d]))
                    # = min(1, exp(-0.5*(xi^2 - u_curr[d]^2)))
                    log_alpha = -0.5 * (xi**2 - u_curr[d] ** 2)
                    if np.log(self._random.random()) < log_alpha:
                        u_prop[d] = xi

                # Accept the joint proposal if it stays in the conditional region
                x_prop = self.transform.u_to_x(u_prop, marg)
                G_prop, _ = self._lsf(x_prop.reshape(-1, 1))
                g_prop = float(G_prop[0, 0])

                if g_prop <= threshold:
                    u_curr = u_prop
                    g_curr = g_prop

                u_out[:, idx] = u_curr
                G_out[idx] = g_curr
                idx += 1

        return u_out, G_out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
