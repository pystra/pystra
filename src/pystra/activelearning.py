# -*- coding: utf-8 -*-
"""Active learning reliability analysis.

This module implements surrogate-based active learning for structural
reliability, following the AK-MCS framework (Echard et al., 2011).

The core loop:
1. Build an initial experimental design (LHS in standard normal space)
2. Evaluate the true limit state function at those points
3. Fit a surrogate model (Kriging or PCE)
4. Evaluate the surrogate on a large Monte Carlo candidate population
5. Select the most informative point via a learning function (U or EFF)
6. Evaluate the true LSF at that point and add to the design
7. Check convergence; repeat from step 3 if not converged
8. Estimate Pf from the final surrogate classification

References
----------
Echard, B., Gayton, N., & Lemaire, M. (2011). AK-MCS: An active learning
reliability method combining Kriging and Monte Carlo Simulation.
*Structural Safety*, 33(2), 145–154.

Bichon, B. J., Eldred, M. S., Swiler, L. P., Mahadevan, S., & McFarland,
J. M. (2008). Efficient global reliability analysis for nonlinear implicit
performance functions. *AIAA Journal*, 46(10), 2459–2468.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm as scipy_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

from .analysis import AnalysisObject


# ======================================================================
# Surrogate interface and implementations
# ======================================================================


class Surrogate(ABC):
    """Abstract base class for surrogate models used in active learning."""

    @abstractmethod
    def fit(self, X, y):
        """Fit the surrogate to training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training inputs in physical space.
        y : ndarray, shape (n_samples,)
            Training LSF values.
        """

    @abstractmethod
    def predict(self, X):
        """Predict mean and standard deviation at new points.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Prediction points in physical space.

        Returns
        -------
        mean : ndarray, shape (n_samples,)
        std : ndarray, shape (n_samples,)
        """


class KrigingSurrogate(Surrogate):
    """Gaussian Process Regression (Kriging) surrogate.

    Wraps scikit-learn's ``GaussianProcessRegressor`` with a Matérn 5/2
    kernel, which is a common choice for structural reliability problems.

    Inputs are standardised (zero mean, unit variance) before fitting
    to handle variables with very different scales (e.g. a beam problem
    with b ~ 0.15 m and E ~ 30000 MPa).

    Parameters
    ----------
    n_restarts : int, optional
        Number of random restarts for kernel hyperparameter optimisation
        (default 5).
    noise : float, optional
        Regularisation nugget added to the diagonal (default 1e-10).
    """

    def __init__(self, n_restarts=5, noise=1e-10):
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            alpha=noise,
            normalize_y=True,
        )
        self._X_mean = None
        self._X_std = None

    def fit(self, X, y):
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        # Avoid division by zero for constant features
        self._X_std[self._X_std < 1e-12] = 1.0
        X_scaled = (X - self._X_mean) / self._X_std
        self._gpr.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = (X - self._X_mean) / self._X_std
        mean, std = self._gpr.predict(X_scaled, return_std=True)
        # Clamp std to small positive value to avoid division by zero
        std = np.maximum(std, 1e-12)
        return mean, std


class PCESurrogate(Surrogate):
    """Polynomial Chaos Expansion surrogate.

    Uses ``chaospy`` for polynomial basis construction and least-squares
    regression fitting via the point-collocation method.

    Prediction variance is estimated by leave-one-out cross-validation
    error, following the approach used in UQlab for PCE-based active
    learning.

    Parameters
    ----------
    degree : int, optional
        Maximum total polynomial degree (default 3).

    Notes
    -----
    Requires the ``chaospy`` package (``pip install chaospy``).
    """

    def __init__(self, degree=3):
        try:
            import chaospy
        except ImportError:
            raise ImportError(
                "The 'chaospy' package is required for PCE surrogate. "
                "Install it with: pip install chaospy"
            )
        self.degree = degree
        self._poly = None
        self._coeffs = None
        self._loo_mse = None
        self._X_train = None
        self._y_train = None

    def fit(self, X, y):
        import chaospy

        n_samples, n_dim = X.shape
        self._X_train = X.copy()
        self._y_train = y.copy()

        # Build orthogonal polynomial basis using a Normal distribution
        # (we work in physical space but the basis just needs to span the data)
        distributions = [chaospy.Normal(0, 1) for _ in range(n_dim)]
        joint = chaospy.J(*distributions)

        expansion = chaospy.generate_expansion(self.degree, joint)

        # Evaluate basis polynomials at training points
        # chaospy expects shape (n_dim, n_samples)
        design_matrix = expansion(*X.T)  # shape (n_terms, n_samples)
        A = design_matrix.T  # shape (n_samples, n_terms)

        # Least-squares fit
        self._coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self._poly = expansion
        self._design_matrix_train = A

        # Leave-one-out cross-validation for variance estimate
        self._compute_loo_error(A, y)

    def _compute_loo_error(self, A, y):
        """Compute LOO cross-validation error for variance estimation."""
        n = len(y)
        try:
            H = A @ np.linalg.pinv(A.T @ A) @ A.T
            y_pred = A @ self._coeffs
            residuals = y - y_pred
            # LOO residual: e_i / (1 - h_ii)
            h_diag = np.diag(H)
            h_diag = np.minimum(h_diag, 1.0 - 1e-10)  # avoid division by zero
            loo_residuals = residuals / (1.0 - h_diag)
            self._loo_mse = np.mean(loo_residuals**2)
        except np.linalg.LinAlgError:
            # Fallback: use residual variance
            y_pred = A @ self._coeffs
            self._loo_mse = np.var(y - y_pred)

    def predict(self, X):
        if self._poly is None:
            raise RuntimeError("Surrogate has not been fitted yet")

        design_matrix = self._poly(*X.T)  # (n_terms, n_samples)
        mean = design_matrix.T @ self._coeffs

        # Use LOO MSE as a spatially-uniform variance estimate
        std = np.full_like(mean, np.sqrt(max(self._loo_mse, 1e-12)))
        return mean, std


# ======================================================================
# Learning functions
# ======================================================================


def learning_U(mean, std):
    r"""U-function (Echard et al., 2011).

    .. math::
        U(\mathbf{x}) = \frac{|\hat{\mu}(\mathbf{x})|}
                              {\hat{\sigma}(\mathbf{x})}

    Points with the smallest U are closest to the limit state surface
    relative to the prediction uncertainty — they are the most
    informative to add.

    Parameters
    ----------
    mean, std : ndarray
        Surrogate predictions (mean and standard deviation).

    Returns
    -------
    U : ndarray
        Learning function values.  **Smaller is better** (more
        informative).
    best_idx : int
        Index of the most informative candidate.
    converged : bool
        ``True`` if ``min(U) >= 2``, meaning no candidate lies within
        2 standard deviations of the limit state.
    """
    U = np.abs(mean) / std
    best_idx = int(np.argmin(U))
    converged = U[best_idx] >= 2.0
    return U, best_idx, converged


def learning_EFF(mean, std):
    r"""Expected Feasibility Function (Bichon et al., 2008).

    .. math::
        \text{EFF}(\mathbf{x}) = \mu \bigl[2\Phi\!\bigl(\frac{-|\mu|}
        {\sigma}\bigr) - \Phi\!\bigl(\frac{-\varepsilon - \mu}{\sigma}\bigr)
        - \Phi\!\bigl(\frac{-\varepsilon + \mu}{\sigma}\bigr)\bigr]
        + \sigma \bigl[2\varphi\!\bigl(\frac{-|\mu|}{\sigma}\bigr)
        - \varphi\!\bigl(\frac{-\varepsilon - \mu}{\sigma}\bigr)
        - \varphi\!\bigl(\frac{-\varepsilon + \mu}{\sigma}\bigr)\bigr]

    where :math:`\varepsilon = 2\sigma`.

    Parameters
    ----------
    mean, std : ndarray
        Surrogate predictions.

    Returns
    -------
    EFF : ndarray
        Learning function values.  **Larger is better** (more
        informative).
    best_idx : int
        Index of the most informative candidate.
    converged : bool
        ``True`` if ``max(EFF) < 1e-3``.
    """
    eps = 2.0 * std
    u_val = np.abs(mean) / std

    term1 = mean * (
        2.0 * scipy_norm.cdf(-u_val)
        - scipy_norm.cdf((-eps - mean) / std)
        - scipy_norm.cdf((-eps + mean) / std)
    )
    term2 = std * (
        2.0 * scipy_norm.pdf(-u_val)
        - scipy_norm.pdf((-eps - mean) / std)
        - scipy_norm.pdf((-eps + mean) / std)
    )
    EFF = term1 + term2
    EFF = np.abs(EFF)  # numerical safety

    best_idx = int(np.argmax(EFF))
    converged = EFF[best_idx] < 1e-3
    return EFF, best_idx, converged


# Registry mapping keyword → (function, default convergence)
_LEARNING_FUNCTIONS = {
    "U": learning_U,
    "EFF": learning_EFF,
}

# Registry mapping keyword → surrogate class
_SURROGATES = {
    "kriging": KrigingSurrogate,
    "pce": PCESurrogate,
}

# Default learning function per surrogate type
_DEFAULT_LF = {
    "kriging": "U",
    "pce": "U",
}


# ======================================================================
# Active Learning analysis class
# ======================================================================


class ActiveLearning(AnalysisObject):
    r"""Active learning reliability analysis (AK-MCS).

    Uses a surrogate model with adaptive sampling to estimate the
    probability of failure with a small number of true limit state
    function evaluations.

    Parameters
    ----------
    stochastic_model : StochasticModel
    limit_state : LimitState
    analysis_options : AnalysisOptions
    surrogate : str, optional
        Surrogate model type: ``"kriging"`` (default) or ``"pce"``.
    learning_function : str or None, optional
        Learning function: ``"U"`` or ``"EFF"``.  If ``None``, defaults
        to ``"U"`` for Kriging, ``"U"`` for PCE.
    n_initial : int or None, optional
        Number of initial experimental design points.  If ``None``,
        uses ``max(10, 2 * nrv)``.
    n_candidates : int, optional
        Size of the Monte Carlo candidate population evaluated on the
        surrogate at each iteration (default 10000).
    max_iterations : int, optional
        Maximum number of enrichment iterations (default 200).
    conv_threshold : float, optional
        Relative convergence threshold for beta stability,
        ``|Δβ/β| < threshold`` (default 0.01).
    conv_iters : int, optional
        Number of consecutive iterations that must satisfy convergence
        (default 2).
    surrogate_kwargs : dict or None, optional
        Extra keyword arguments passed to the surrogate constructor.

    Attributes
    ----------
    Pf : float
        Estimated probability of failure.
    beta : float
        Reliability index.
    n_evals : int
        Total number of true LSF evaluations.
    surrogate_model : Surrogate
        The fitted surrogate after convergence.
    history : dict
        Convergence history (beta, Pf, learning function values per
        iteration).
    """

    def __init__(
        self,
        analysis_options=None,
        limit_state=None,
        stochastic_model=None,
        surrogate="kriging",
        learning_function=None,
        n_initial=None,
        n_candidates=10_000,
        max_iterations=200,
        conv_threshold=0.01,
        conv_iters=2,
        surrogate_kwargs=None,
    ):
        super().__init__(
            analysis_options=analysis_options,
            limit_state=limit_state,
            stochastic_model=stochastic_model,
        )

        # Surrogate setup
        surrogate_key = surrogate.lower()
        if surrogate_key not in _SURROGATES:
            raise ValueError(
                f"Unknown surrogate '{surrogate}'. "
                f"Available: {list(_SURROGATES.keys())}"
            )
        self._surrogate_key = surrogate_key
        self._surrogate_kwargs = surrogate_kwargs or {}

        # Learning function
        if learning_function is None:
            learning_function = _DEFAULT_LF[surrogate_key]
        lf_key = learning_function.upper()
        if lf_key not in _LEARNING_FUNCTIONS:
            raise ValueError(
                f"Unknown learning function '{learning_function}'. "
                f"Available: {list(_LEARNING_FUNCTIONS.keys())}"
            )
        self._lf_key = lf_key
        self._lf_func = _LEARNING_FUNCTIONS[lf_key]

        # AL parameters
        self._n_initial = n_initial
        self.n_candidates = n_candidates
        self.max_iterations = max_iterations
        self.conv_threshold = conv_threshold
        self.conv_iters = conv_iters

        # Results (populated by run())
        self.Pf = None
        self.beta = None
        self.cov = None
        self.n_evals = 0
        self.surrogate_model = None
        self.history = {
            "beta": [],
            "Pf": [],
            "lf_best": [],
            "n_evals": [],
        }

    def run(self):
        """Execute the active learning reliability analysis."""
        self.results_valid = True
        self.init_run()

        nrv = self.model.n_marg
        marg = self.model.marginal_distributions

        n_initial = self._n_initial or max(10, 2 * nrv)

        # ==============================================================
        # Step 1: Generate initial experimental design (LHS in u-space)
        # ==============================================================
        u_doe = self._lhs(nrv, n_initial)
        x_doe = np.array([self.transform.u_to_x(u_doe[i], marg) for i in range(n_initial)])
        g_doe = self._eval_lsf(x_doe)
        self.n_evals = n_initial

        if self.options.getPrintOutput():
            print(f" Initial design: {n_initial} points, {nrv} variables")

        # ==============================================================
        # Step 2: Generate candidate population (MCS in u-space)
        # ==============================================================
        u_cand = np.random.randn(self.n_candidates, nrv)
        x_cand = np.array(
            [self.transform.u_to_x(u_cand[i], marg) for i in range(self.n_candidates)]
        )

        # ==============================================================
        # Step 3: Iterative enrichment loop
        # ==============================================================
        surr = _SURROGATES[self._surrogate_key](**self._surrogate_kwargs)
        n_consec = 0
        beta_prev = None

        for iteration in range(1, self.max_iterations + 1):
            # Fit surrogate
            surr.fit(x_doe, g_doe)

            # Predict on candidate population
            g_mean, g_std = surr.predict(x_cand)

            # Estimate Pf from surrogate sign
            n_fail = np.sum(g_mean <= 0)
            Pf = n_fail / self.n_candidates
            if 0 < Pf < 1:
                beta = float(-scipy_norm.ppf(Pf))
            elif Pf <= 0:
                beta = np.inf
            else:
                beta = -np.inf

            # Evaluate learning function
            lf_vals, best_idx, lf_converged = self._lf_func(g_mean, g_std)

            # Record history
            self.history["beta"].append(beta)
            self.history["Pf"].append(Pf)
            self.history["lf_best"].append(
                float(np.min(lf_vals)) if self._lf_key == "U" else float(np.max(lf_vals))
            )
            self.history["n_evals"].append(self.n_evals)

            if self.options.getPrintOutput():
                lf_val = self.history["lf_best"][-1]
                print(
                    f" Iter {iteration:3d}: beta = {beta:.4f}, "
                    f"Pf = {Pf:.4e}, LF = {lf_val:.4f}, "
                    f"N_eval = {self.n_evals}"
                )

            # Check convergence: LF criterion AND beta stability
            beta_stable = False
            if beta_prev is not None and np.isfinite(beta) and beta != 0:
                rel_change = abs(beta - beta_prev) / abs(beta)
                beta_stable = rel_change < self.conv_threshold

            if lf_converged and beta_stable:
                n_consec += 1
            else:
                n_consec = 0

            if n_consec >= self.conv_iters:
                if self.options.getPrintOutput():
                    print(f" Converged after {iteration} iterations")
                break

            beta_prev = beta

            # Enrich: add the best candidate to the experimental design
            x_new = x_cand[best_idx : best_idx + 1]
            g_new = self._eval_lsf(x_new)
            self.n_evals += 1

            x_doe = np.vstack([x_doe, x_new])
            g_doe = np.append(g_doe, g_new)

        else:
            if self.options.getPrintOutput():
                print(
                    f" Warning: max iterations ({self.max_iterations}) "
                    f"reached without convergence"
                )

        # ==============================================================
        # Step 4: Final Pf estimate from converged surrogate
        # ==============================================================
        surr.fit(x_doe, g_doe)
        g_mean_final, _ = surr.predict(x_cand)
        n_fail = np.sum(g_mean_final <= 0)
        self.Pf = float(n_fail / self.n_candidates)
        if 0 < self.Pf < 1:
            self.beta = float(-scipy_norm.ppf(self.Pf))
        elif self.Pf <= 0:
            self.beta = np.inf
        else:
            self.beta = -np.inf

        # CoV estimate from binomial proportion
        if 0 < self.Pf < 1:
            self.cov = float(np.sqrt((1 - self.Pf) / (self.n_candidates * self.Pf)))
        else:
            self.cov = 0.0

        self.surrogate_model = surr

        if self.options.getPrintOutput():
            self.showResults()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lhs(self, ndim, n_samples):
        """Generate Latin Hypercube Samples in standard normal space.

        Parameters
        ----------
        ndim : int
            Number of dimensions.
        n_samples : int
            Number of samples.

        Returns
        -------
        samples : ndarray, shape (n_samples, ndim)
            Samples in standard normal space.
        """
        result = np.empty((n_samples, ndim))
        for d in range(ndim):
            perm = np.random.permutation(n_samples)
            # Stratified uniform samples mapped to standard normal
            u = (perm + np.random.rand(n_samples)) / n_samples
            result[:, d] = scipy_norm.ppf(u)
        return result

    def _eval_lsf(self, x):
        """Evaluate the true LSF at physical-space points.

        Parameters
        ----------
        x : ndarray, shape (n_points, nrv)
            Points in physical space (row-wise).

        Returns
        -------
        G : ndarray, shape (n_points,)
            Limit state function values.
        """
        # LimitState.evaluate_lsf expects shape (nrv, n_points)
        x_T = x.T
        G, _ = self.limitstate.evaluate_lsf(
            x_T, self.model, self.options, "no"
        )
        return G.ravel()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def getBeta(self):
        """Return the reliability index."""
        return self.beta

    def getFailure(self):
        """Return the probability of failure."""
        return self.Pf

    def showResults(self):
        """Print a summary of active learning results."""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        n_hyph = self.N_HYPH
        print("")
        print("=" * n_hyph)
        print("")
        print(" RESULTS FROM RUNNING ACTIVE LEARNING RELIABILITY")
        print("")
        print(f" Surrogate model:              {self._surrogate_key}")
        print(f" Learning function:            {self._lf_key}")
        print(f" Reliability index beta:       {self.beta:.6f}")
        print(f" Failure probability:          {self.Pf:.6e}")
        print(f" Coefficient of variation:     {self.cov:.4f}")
        print(f" Total LSF evaluations:        {self.n_evals}")
        print(f" Enrichment iterations:        {len(self.history['beta'])}")
        print("")
        print("=" * n_hyph)
        print("")
