# -*- coding: utf-8 -*-
"""Line Sampling reliability analysis."""

import numpy as np
from scipy import optimize
from scipy.stats import norm as scipy_norm

from .analysis import AnalysisObject
from .form import Form


class LineSampling(AnalysisObject):
    r"""Line Sampling (LS) reliability analysis.

    Line Sampling exploits the important direction :math:`\boldsymbol{\alpha}`
    obtained from FORM.  For each of *N* random samples drawn uniformly in the
    (n-1)-dimensional hyperplane perpendicular to
    :math:`\boldsymbol{\alpha}`, a one-dimensional root-finding problem
    locates the limit-state surface along the parallel line.

    The failure-probability estimate is

    .. math::

       \hat{p}_f = \frac{1}{N} \sum_{i=1}^{N} \Phi(-c_i)

    where :math:`c_i` is the signed distance from the foot-point of sample
    *i* to the limit-state surface along :math:`\boldsymbol{\alpha}`, and
    :math:`\Phi` is the standard normal CDF.

    Parameters
    ----------
    stochastic_model : StochasticModel
    limit_state : LimitState
    analysis_options : AnalysisOptions
    form : Form, optional
        A pre-computed FORM result.  If ``None``, FORM is run automatically
        to obtain the important direction :math:`\boldsymbol{\alpha}` and
        the initial guess for the root search.

    Attributes
    ----------
    Pf : float
        Estimated probability of failure.
    beta : float
        Reliability index :math:`\beta = -\Phi^{-1}(p_f)`.
    cov : float
        Estimated coefficient of variation of :math:`\hat{p}_f`.
    alpha : ndarray, shape (nrv,)
        Important direction used in the analysis.
    n_samples : int
        Number of lines (samples) used.

    References
    ----------
    Koutsourelakis, P. S., Pradlwarter, H. J., & Schuëller, G. I. (2004).
    Reliability of structures in high dimensions, Part I: algorithms and
    applications. *Probabilistic Engineering Mechanics*, 19(4), 409–417.
    """

    def __init__(
        self,
        analysis_options=None,
        limit_state=None,
        stochastic_model=None,
        form=None,
    ):
        super().__init__(
            analysis_options=analysis_options,
            limit_state=limit_state,
            stochastic_model=stochastic_model,
        )
        self.form = form
        self.nrv = self.model.getLenMarginalDistributions()
        self.alpha = None
        self.Pf = None
        self.beta = None
        self.cov = None
        self.n_samples = None
        self._pf_contributions = None

    def run(self):
        """Execute the Line Sampling analysis."""
        self.results_valid = True
        self.init_run()

        marg = self.model.getMarginalDistributions()

        # Obtain important direction from FORM
        if self.form is None:
            _form = Form(
                stochastic_model=self.model,
                limit_state=self.limitstate,
                analysis_options=self.options,
            )
            _form.run()
            self.form = _form

        # alpha: unit vector pointing toward the failure region in u-space
        alpha = self.form.getAlpha()  # shape (nrv,)
        beta_form = float(self.form.getBeta())
        self.alpha = alpha

        N = self.options.getSamples()
        n = self.nrv

        # Draw N samples in standard normal space
        u_samples = np.random.randn(n, N)  # (n, N)

        # Project out the component along alpha to get perpendicular components
        # v_i = u_i - (u_i · alpha) * alpha
        alpha_col = alpha.reshape(-1, 1)  # (n, 1)
        u_perp = u_samples - alpha_col * (alpha_col.T @ u_samples)  # (n, N)

        # For each line, find c_i such that g(v_i + c_i * alpha) = 0
        c_values = np.empty(N)
        for i in range(N):
            c_values[i] = self._find_line_intersection(
                u_perp[:, i], alpha, beta_form, marg
            )

        # Probability contributions Phi(-c_i): probability that a point on
        # line i (drawn from N(0,1) along alpha) lies in the failure region.
        pf_contribs = scipy_norm.cdf(-c_values)
        self._pf_contributions = pf_contribs

        self.Pf = float(np.mean(pf_contribs))
        self.n_samples = N

        if 0.0 < self.Pf < 1.0:
            self.beta = float(-scipy_norm.ppf(self.Pf))
            self.cov = float(np.std(pf_contribs) / (np.sqrt(N) * self.Pf))
        elif self.Pf <= 0.0:
            self.beta = np.inf
            self.cov = np.inf
        else:
            self.beta = -np.inf
            self.cov = np.inf

        if self.options.getPrintOutput():
            self.showResults()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_g_at_c(self, c, v, alpha, marg):
        """Evaluate g(v + c * alpha) and return the scalar LSF value."""
        u_pt = v + c * alpha
        x_pt = self.transform.u_to_x(u_pt, marg)
        G, _ = self.limitstate.evaluate_lsf(
            x_pt.reshape(-1, 1), self.model, self.options, "no"
        )
        return float(G[0, 0])

    def _find_line_intersection(self, v, alpha, c_init, marg):
        """Find c such that g(v + c * alpha) = 0 along the important direction.

        Scans the range [-c_max, c_max] for a sign change, then refines with
        Brent's method.  Returns a large positive value when the line lies
        entirely in the safe region, or a large negative value when it lies
        entirely in the failure region.
        """
        c_max = max(abs(c_init) * 3.0 + 5.0, 15.0)
        c_scan = np.linspace(-c_max, c_max, 40)

        g_scan = np.empty(len(c_scan))
        for j, cj in enumerate(c_scan):
            try:
                g_scan[j] = self._eval_g_at_c(cj, v, alpha, marg)
            except Exception:
                g_scan[j] = np.nan

        # Collect all sign-change intervals, preferring safe→failure crossings
        bracket_lo, bracket_hi = None, None
        for j in range(len(c_scan) - 1):
            g1, g2 = g_scan[j], g_scan[j + 1]
            if np.isnan(g1) or np.isnan(g2):
                continue
            if g1 * g2 <= 0:
                if g1 > 0 >= g2:
                    # Safe→failure crossing: this is the physically relevant one
                    bracket_lo, bracket_hi = c_scan[j], c_scan[j + 1]
                    break
                if bracket_lo is None:
                    # failure→safe crossing; keep as fallback
                    bracket_lo, bracket_hi = c_scan[j], c_scan[j + 1]

        if bracket_lo is None:
            # No sign change found
            valid = g_scan[~np.isnan(g_scan)]
            if len(valid) > 0 and valid[0] < 0:
                return float(c_scan[0])  # entirely in failure region → Phi(-c)≈1
            return float(c_scan[-1])  # entirely in safe region → Phi(-c)≈0

        try:
            c_root = optimize.brentq(
                lambda c: self._eval_g_at_c(c, v, alpha, marg),
                min(bracket_lo, bracket_hi),
                max(bracket_lo, bracket_hi),
                xtol=1e-6,
                maxiter=50,
            )
        except Exception:
            c_root = 0.5 * (bracket_lo + bracket_hi)

        return float(c_root)

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
        """Print a summary of Line Sampling results to the console."""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        n_hyph = self.N_HYPH
        print("")
        print("=" * n_hyph)
        print("")
        print(" RESULTS FROM RUNNING LINE SAMPLING")
        print("")
        print(f" Reliability index beta:        {self.beta:.6f}")
        print(f" Failure probability:           {self.Pf:.6e}")
        print(f" Coefficient of variation:      {self.cov:.4f}")
        print(f" Number of lines (samples):     {self.n_samples}")
        print("")
        print("=" * n_hyph)
        print("")
