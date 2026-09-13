"""Line Sampling reliability analysis."""

import numpy as np
from scipy import optimize
from scipy.special import log_ndtr, logsumexp, ndtri_exp

from .analysis import AnalysisObject, _check_rng, _generator
from .form import FORM
from ._form_reuse import _check_form, _check_coordinates, _FORMReuse
from ..options import FORMOptions, SimulationOptions
from ..errors import AnalysisError
from ..results import FORMResult, SimulationResult

__all__ = ["LineSampling"]


class LineSampling(_FORMReuse, AnalysisObject):
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
    model : StochasticModel
    limit_state : LimitState
    options : SimulationOptions, optional
        ``n_samples`` is the number of lines.
    form : FORM, optional
        A completed FORM analysis. If ``None``, :meth:`run` first runs FORM,
        with this analysis's block size and transformation, to obtain the
        important direction :math:`\boldsymbol{\alpha}` and the initial
        guess for the root search.
    rng : int, numpy.random.Generator or None, optional
        Random source; NumPy's global generator is not used. A seed recreates
        the same stream on every run, a generator advances its own state, and
        None draws fresh entropy.

    Notes
    -----
    :meth:`run` returns a :class:`~pystra.results.SimulationResult`; its
    diagnostics include the important ``direction``. The line search is
    restricted to normal coordinates in [-37, 37], including correlated
    marginal coordinates for Nataf transformations. If that restriction
    truncates the scan and no crossing is found, an AnalysisError is raised;
    crossings beyond the representable scan are not assigned a probability.

    References
    ----------
    Koutsourelakis, P. S., Pradlwarter, H. J., & Schuëller, G. I. (2004).
    Reliability of structures in high dimensions, Part I: algorithms and
    applications. *Probabilistic Engineering Mechanics*, 19(4), 409–417.
    """

    _options_type = SimulationOptions

    def __init__(self, model, limit_state, *, options=None, form=None, rng=None):
        super().__init__(model, limit_state, options)
        self.rng = _check_rng(rng)
        self.options._require_defaults(
            "LineSampling", ("target_cov", "sampling_std", "bins")
        )
        if form is not None and not isinstance(form, FORM):
            raise TypeError("form must be a FORM analysis")
        self.form = form
        self._nrv = self.model.get_len_marginal_distributions()
        self._alpha = None
        self._Pf = None
        self._beta = None
        self._cov = None
        self._n_samples = None
        self._pf_contributions = None

    def run(self):
        """Run line sampling and return a :class:`SimulationResult`."""
        self._results_valid = False
        self._Pf = self._beta = self._cov = None
        self.init_run()

        marg = self.model.get_marginal_distributions()
        self._nrv = len(marg)

        # Obtain important direction from FORM
        if self._supplied_form is None:
            _form = FORM(
                self.model,
                self.limit_state,
                on_failure="return",
                options=FORMOptions(
                    block_size=self.options.block_size,
                    transform=self.options.transform,
                    rosenblatt_order=self.options.rosenblatt_order,
                ),
            )
            _form.run()
            self._form = _form
        _check_form(self.form, self.model, self.limit_state)
        _check_coordinates(self.form, self.transform)

        # alpha: unit vector pointing toward the failure region in u-space
        alpha = self.form._alpha[0]  # shape (nrv,)
        beta_form = self.form._beta
        self._alpha = alpha

        N = self.options.n_samples
        n = self._nrv

        # Draw N samples in standard normal space
        u_samples = _generator(self.rng).standard_normal((n, N))

        # Project out the component along alpha to get perpendicular components
        # v_i = u_i - (u_i · alpha) * alpha
        alpha_col = alpha.reshape(-1, 1)  # (n, 1)
        u_perp = u_samples - alpha_col * (alpha_col.T @ u_samples)  # (n, N)

        # For each line, find c_i such that g(v_i + c_i * alpha) = 0
        c_values = np.empty(N)
        for i in range(N):
            try:
                c_values[i] = self._find_line_intersection(
                    u_perp[:, i], alpha, beta_form, marg
                )
            except AnalysisError as error:
                raise AnalysisError(f"Line {i + 1}: {error}") from error

        # Probability contributions Phi(-c_i): probability that a point on
        # line i (drawn from N(0,1) along alpha) lies in the failure region.
        log_contribs = log_ndtr(-c_values)
        self._pf_contributions = np.exp(log_contribs)
        log_pf = logsumexp(log_contribs) - np.log(N)
        self._Pf = float(np.exp(log_pf))
        self._n_samples = N
        self._beta = float(-ndtri_exp(log_pf))
        if np.isfinite(log_pf) and log_pf < 0:
            # Normalize before squaring: probabilities below 1e-162 have
            # representable relative dispersion but their squares underflow.
            relative = np.exp(log_contribs - log_pf)
            self._cov = float(np.std(relative) / np.sqrt(N))
        else:
            self._cov = np.inf

        result = SimulationResult(
            method="LineSampling",
            status="completed",
            message="Sampling completed",
            n_limit_state_evaluations=self._n_evaluations,
            variable_names=tuple(self.model.get_variables()),
            failure_probability=self._Pf,
            beta=float(self._beta),
            coefficient_of_variation=self._cov,
            n_samples=N,
            options=self.options,
            diagnostics={
                "form": FORMResult.from_analysis(self.form),
                "direction": alpha,
            },
        )
        self._results_valid = True
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_g_at_c(self, c, v, alpha, marg):
        """Evaluate g(v + c * alpha) and return the scalar LSF value."""
        u_pt = v + c * alpha
        x_pt = self.transform.u_to_x(u_pt, marg)
        G, _ = self._lsf(x_pt.reshape(-1, 1))
        return float(G[0, 0])

    def _find_line_intersection(self, v, alpha, c_init, marg):
        """Find c such that g(v + c * alpha) = 0 along the important direction.

        Scans the range [-c_max, c_max] for a sign change, then refines with
        Brent's method.  Returns a large positive value when the line lies
        entirely in the safe region, or a large negative value when it lies
        entirely in the failure region.
        """
        c_max = max(abs(c_init) * 3.0 + 5.0, 15.0)
        # Intersect the scan with every coordinate's representable normal
        # tail. Also constrain correlated marginal coordinates for Nataf.
        offsets, directions = np.asarray(v), np.asarray(alpha)
        factor = getattr(self.transform, "inv_T", None)
        if factor is not None:
            offsets = np.concatenate((offsets, factor @ v))
            directions = np.concatenate((directions, factor @ alpha))
        moving = directions != 0
        if np.any(np.abs(offsets[~moving]) > 37.0):
            raise AnalysisError(
                "Line lies outside the supported normal range [-37, 37]"
            )
        ends1 = (-37.0 - offsets[moving]) / directions[moving]
        ends2 = (37.0 - offsets[moving]) / directions[moving]
        lo = max(-c_max, float(np.max(np.minimum(ends1, ends2))))
        hi = min(c_max, float(np.min(np.maximum(ends1, ends2))))
        if lo >= hi:
            raise AnalysisError(
                "Line does not intersect the supported normal range [-37, 37]"
            )
        clipped = lo > -c_max or hi < c_max
        # Stay inside the boundary despite rounding in v + c * alpha.
        c_scan = np.linspace(np.nextafter(lo, hi), np.nextafter(hi, lo), 40)

        g_scan = np.empty(len(c_scan))
        for j, cj in enumerate(c_scan):
            g_scan[j] = self._eval_g_at_c(cj, v, alpha, marg)

        # Collect all sign-change intervals, preferring safe→failure crossings
        bracket_lo, bracket_hi = None, None
        for j in range(len(c_scan) - 1):
            g1, g2 = g_scan[j], g_scan[j + 1]
            if g1 == 0 or g2 == 0 or np.signbit(g1) != np.signbit(g2):
                if g1 > 0 >= g2:
                    # Safe→failure crossing: this is the physically relevant one
                    bracket_lo, bracket_hi = c_scan[j], c_scan[j + 1]
                    break
                if bracket_lo is None:
                    # failure→safe crossing; keep as fallback
                    bracket_lo, bracket_hi = c_scan[j], c_scan[j + 1]

        if bracket_lo is None:
            # A truncated search cannot classify the unsearched tail.
            if clipped:
                raise AnalysisError(
                    "No line intersection within the supported normal range [-37, 37]"
                )
            if np.all(g_scan < 0):
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
        except Exception as error:
            raise AnalysisError(
                f"Line intersection did not converge: {error}"
            ) from error

        return float(c_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
