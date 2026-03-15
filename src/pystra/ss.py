# -*- coding: utf-8 -*-
"""Subset Simulation reliability analysis."""

import numpy as np
from scipy.stats import norm as scipy_norm

from .analysis import AnalysisObject


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
    component-wise in standard-normal space.

    Parameters
    ----------
    stochastic_model : StochasticModel
    limit_state : LimitState
    analysis_options : AnalysisOptions
    p0 : float, optional
        Target conditional failure probability per subset level (default 0.1).
    proposal_sigma : float, optional
        Half-width of the uniform proposal kernel used in MMH, measured in
        standard-deviation units of the standard-normal space (default 1.0).

    Attributes
    ----------
    Pf : float
        Estimated probability of failure.
    beta : float
        Reliability index :math:`\beta = -\Phi^{-1}(p_f)`.
    cov : float
        Estimated coefficient of variation of :math:`\hat{p}_f` (ignoring
        Markov-chain correlations; a lower bound on the true CoV).
    thresholds : list of float
        Intermediate thresholds :math:`y_1, \ldots, y_m` used at each level.
    conditional_probs : list of float
        Conditional failure probability estimate at each level.
    n_levels : int
        Total number of subset levels (including level 0).

    Notes
    -----
    The number of samples per level is taken from
    ``analysis_options.getSamples()``.

    References
    ----------
    Au, S. K., & Beck, J. L. (2001).  Estimation of small failure
    probabilities in high dimensions by subset simulation.
    *Probabilistic Engineering Mechanics*, 16(4), 263–277.
    """

    def __init__(
        self,
        analysis_options=None,
        limit_state=None,
        stochastic_model=None,
        p0=0.1,
        proposal_sigma=1.0,
    ):
        super().__init__(
            analysis_options=analysis_options,
            limit_state=limit_state,
            stochastic_model=stochastic_model,
        )
        if not (0.0 < p0 < 1.0):
            raise ValueError(f"p0 must be in (0, 1); got {p0}")
        self.p0 = p0
        self.proposal_sigma = proposal_sigma
        self.Pf = None
        self.beta = None
        self.cov = None
        self.thresholds = []
        self.conditional_probs = []
        self.n_levels = None

    def run(self):
        """Execute the Subset Simulation analysis."""
        self.results_valid = True
        self.init_run()

        nrv = self.model.getLenMarginalDistributions()
        marg = self.model.getMarginalDistributions()
        N = self.options.getSamples()
        p0 = self.p0

        self.thresholds = []
        self.conditional_probs = []

        # ---------------------------------------------------------------
        # Level 0: direct Monte Carlo from the prior N(0, I)
        # ---------------------------------------------------------------
        u = np.random.randn(nrv, N)
        G = self._eval_g_batch(u, marg)

        # y_1 = p0-th quantile of G (the p0 fraction closest to failure G<0)
        y = float(np.percentile(G, p0 * 100.0))

        if y <= 0.0:
            # Threshold already at or below failure: level-0 MC suffices
            Pf = float(np.sum(G <= 0.0)) / N
            self.thresholds.append(0.0)
            self.conditional_probs.append(Pf)
            self.n_levels = 1
            self._finalise(Pf, N)
            return

        self.thresholds.append(y)
        p_lvl = float(np.sum(G <= y)) / N
        self.conditional_probs.append(p_lvl)
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
                self.thresholds.append(0.0)
                self.conditional_probs.append(p_last)
                break

            # Intermediate level
            p_lvl = float(np.sum(G_new <= y_new)) / N
            Pf = Pf * p_lvl
            self.thresholds.append(y_new)
            self.conditional_probs.append(p_lvl)

            # New seeds for the next level
            seed_mask = G_new <= y_new
            seeds_u = u_new[:, seed_mask]
            seeds_G = G_new[seed_mask]

            y = y_new
            level += 1

        self.n_levels = level + 1
        self._finalise(Pf, N)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finalise(self, Pf, N):
        """Store results and compute the CoV estimate."""
        self.Pf = float(Pf)
        if 0.0 < Pf < 1.0:
            self.beta = float(-scipy_norm.ppf(Pf))
        elif Pf <= 0.0:
            self.beta = np.inf
        else:
            self.beta = -np.inf

        # CoV lower bound (γ_j = 0, i.e. ignoring Markov-chain correlations)
        # δ²(Pf) ≈ Σ_j (1 - p_j) / (N * p_j)
        delta_sq = sum(
            (1.0 - p) / (N * p)
            for p in self.conditional_probs
            if p > 0.0
        )
        self.cov = float(np.sqrt(delta_sq)) if delta_sq > 0.0 else 0.0

        if self.options.getPrintOutput():
            self.showResults()

    def _eval_g_batch(self, u, marg):
        """Evaluate the LSF for every column of *u* (shape nrv × N)."""
        N = u.shape[1]
        G = np.empty(N)
        block = self.options.getBlockSize()
        for start in range(0, N, block):
            end = min(start + block, N)
            u_blk = u[:, start:end]
            x_blk = np.empty_like(u_blk)
            for i in range(end - start):
                x_blk[:, i] = self.transform.u_to_x(u_blk[:, i], marg)
            G_blk, _ = self.limitstate.evaluate_lsf(
                x_blk, self.model, self.options, "no"
            )
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
                    xi = u_curr[d] + sigma * np.random.uniform(-1.0, 1.0)
                    # Accept with min(1, phi(xi)/phi(u_curr[d]))
                    # = min(1, exp(-0.5*(xi^2 - u_curr[d]^2)))
                    log_alpha = -0.5 * (xi**2 - u_curr[d] ** 2)
                    if np.log(np.random.rand()) < log_alpha:
                        u_prop[d] = xi

                # Accept the joint proposal if it stays in the conditional region
                x_prop = self.transform.u_to_x(u_prop, marg)
                G_prop, _ = self.limitstate.evaluate_lsf(
                    x_prop.reshape(-1, 1), self.model, self.options, "no"
                )
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

    def getBeta(self):
        """Return the reliability index :math:`\\beta`."""
        return self.beta

    def getFailure(self):
        """Return the probability of failure."""
        return self.Pf

    def showResults(self):
        """Print a summary of Subset Simulation results to the console."""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        n_hyph = 54
        print("")
        print("=" * n_hyph)
        print("")
        print(" RESULTS FROM RUNNING SUBSET SIMULATION")
        print("")
        print(f" Reliability index beta:        {self.beta:.6f}")
        print(f" Failure probability:           {self.Pf:.6e}")
        print(f" Coefficient of variation:      {self.cov:.4f}")
        print(f" Number of subset levels:       {self.n_levels}")
        print("")
        print(" Intermediate results:")
        for j, (y, p) in enumerate(zip(self.thresholds, self.conditional_probs)):
            print(f"   Level {j+1}: threshold = {y:.4f},  p_j = {p:.4f}")
        print("")
        print("=" * n_hyph)
        print("")
