"""Component FORM approximation for series and parallel failure events."""

import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from .analysis import AnalysisObject
from .form import FORM
from ..systems import Component, SeriesSystem, ParallelSystem, ditlevsen_bounds
from ..errors import AnalysisError
from ..options import FORMOptions
from ..results import SystemFORMResult

__all__ = ["SystemFORM"]


class SystemFORM(AnalysisObject):
    """Approximate a series or parallel system using component tangent planes.

    All components use the same complete stochastic model and transformation
    into independent normal coordinates, including the same Rosenblatt order.
    Homogeneous nested series/parallel systems are flattened; mixed topologies
    are rejected. Use original-system Monte Carlo for general topologies.

    Parameters
    ----------
    model : StochasticModel
        Shared variables, constants and dependence model.
    system : SeriesSystem or ParallelSystem
        System failure event.
    options : FORMOptions, optional
        Component FORM settings. DDM gradients must use full model ordering.
    maxpts : int, optional
        Maximum integration points per multivariate normal CDF call.
    abseps, releps : float, optional
        Requested absolute and relative integration tolerances. These are
        numerical targets, not certified error bounds or FORM model errors.

    Notes
    -----
    :meth:`run` returns a :class:`~pystra.results.SystemFORMResult` with each
    component's FORM record, the correlation of the linearized normal scores
    (neither the physical-variable nor the binary-failure correlation), the
    bounds on the linearized event and the pairwise intersections.

    Exact up to integration error for affine limit states in standard normal
    space. Nonlinear components remain first-order approximations, and FORM
    can find a local rather than global design point. Normal integration may
    vary between runs; tight tolerances do not certify rare-tail accuracy.
    """

    _options_type = FORMOptions
    _requires_limit_state = False

    def __init__(
        self,
        model,
        system,
        *,
        options=None,
        maxpts=1000000,
        abseps=1e-10,
        releps=1e-5,
    ):
        if type(system) not in (SeriesSystem, ParallelSystem):
            raise TypeError("SystemFORM requires a series or parallel system")
        super().__init__(model, None, options)
        self.system = system
        leaves = []

        def visit(node):
            if isinstance(node, Component):
                if all(node is not c for c in leaves):
                    leaves.append(node)
            elif type(node) is type(system):
                for child in node.children:
                    visit(child)
            else:
                raise TypeError(
                    "Mixed system topology requires original-system simulation"
                )

        visit(system)
        if len({c.name for c in leaves}) != len(leaves):
            raise ValueError("SystemFORM requires unique component names")
        if isinstance(maxpts, bool) or int(maxpts) != maxpts or maxpts < 1:
            raise ValueError("maxpts must be a positive integer")
        if not all(np.isfinite(t) and t > 0 for t in (abseps, releps)):
            raise ValueError("Integration tolerances must be positive and finite")
        self.components = tuple(leaves)
        self.maxpts, self.abseps, self.releps = int(maxpts), abseps, releps
        self._clear_results()

    def _clear_results(self):
        self._results_valid = False
        self._component_results = {}
        self._Pf = self._beta = self._bounds = None
        self._betas = self._alphas = self._correlation = None
        self._probabilities = self._intersections = None

    def _cdf(self, upper, correlation):
        for i in range(len(upper)):
            for j in range(i):
                if correlation[i, j] == -1 and upper[i] + upper[j] <= 0:
                    return 0.0  # Proven empty event (up to a zero-mass boundary).
        if len(upper) == 1:
            return float(norm.cdf(upper[0]))
        if np.array_equal(correlation, np.eye(len(upper))):
            return float(np.prod(norm.cdf(upper)))
        if np.all(np.abs(correlation[0]) == 1):
            # Every score is the same normal variable, possibly negated.
            positive = correlation[0] == 1
            hi = float(np.min(upper[positive]))
            lo = float(np.max(-upper[~positive])) if np.any(~positive) else -np.inf
            if hi <= lo:
                return 0.0
            if lo > 0:
                return float(norm.sf(lo) - norm.sf(hi))
            return float(norm.cdf(hi) - norm.cdf(lo))
        # Split independent score groups before integration. This also avoids
        # unstable high-dimensional singular CDFs for opposing pairs along
        # independent directions (e.g. the four-branch benchmark).
        remaining = set(range(len(upper)))
        groups = []
        while remaining:
            group = {remaining.pop()}
            frontier = list(group)
            while frontier:
                i = frontier.pop()
                neighbours = {
                    j
                    for j in remaining
                    if abs(correlation[i, j]) > 8 * np.finfo(float).eps
                }
                remaining.difference_update(neighbours)
                group.update(neighbours)
                frontier.extend(neighbours)
            groups.append(sorted(group))
        if len(groups) > 1:
            return float(
                np.prod(
                    [
                        self._cdf(upper[ids], correlation[np.ix_(ids, ids)])
                        for ids in groups
                    ]
                )
            )
        # Exact degeneracies avoid unstable bivariate integration at rho=+/-1.
        if len(upper) == 2:
            rho = correlation[0, 1]
            if rho == 1.0:
                return float(norm.cdf(min(upper)))
            if rho == -1.0:
                lo, hi = -upper[1], upper[0]
                if hi <= lo:
                    return 0.0
                if lo > 0:
                    return float(norm.sf(lo) - norm.sf(hi))
                return float(norm.cdf(hi) - norm.cdf(lo))
            # Integrate the conditional normal directly. Relative tolerance
            # matters here: default absolute CDF tolerances can return zero
            # for representable rare joint probabilities.
            a, b = sorted(upper)
            sigma = np.sqrt((1 - rho) * (1 + rho))
            value, error = quad(
                lambda x: norm.pdf(x) * norm.cdf((b - rho * x) / sigma),
                -np.inf,
                a,
                epsabs=0,
                epsrel=max(self.releps, 1e-12),
                limit=200,
            )
            if error > max(self.abseps / len(self.components), self.releps * value):
                raise AnalysisError(
                    "Bivariate normal integration did not meet tolerance"
                )
            if value == 0:
                raise AnalysisError(
                    "Bivariate probability underflow or unresolved rare tail"
                )
            return float(value)
        value = float(
            multivariate_normal.cdf(
                upper,
                cov=correlation,
                allow_singular=True,
                maxpts=self.maxpts,
                abseps=self.abseps / len(self.components),
                releps=self.releps,
            )
        )
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise AnalysisError("Normal probability integration failed")
        if value == 0:
            raise AnalysisError(
                "Normal integration returned an unresolved zero probability"
            )
        return value

    def run(self):
        """Run each unique component once, then integrate the system event.

        Returns a :class:`SystemFORMResult`.
        """
        self._clear_results()
        records = {}
        for component in self.components:
            form = FORM(self.model, component.as_limit_state(), options=self.options)
            self._component_results[component.name] = form
            try:
                records[component.name] = form.run()
                if form.transform.standard_space != "normal":
                    raise ValueError(
                        "SystemFORM requires independent normal space; select Rosenblatt"
                    )
            except (ValueError, FloatingPointError) as error:
                raise AnalysisError(
                    f"Component '{component.name}' FORM failed: {error}"
                ) from error
            if not form._converged or not np.isfinite(form._beta):
                raise AnalysisError(
                    f"Component '{component.name}' FORM did not converge"
                )

        self._betas = np.array([f._beta for f in self._component_results.values()])
        self._alphas = np.array([f._alpha[0] for f in self._component_results.values()])
        self._correlation = np.clip(self._alphas @ self._alphas.T, -1, 1)
        np.fill_diagonal(self._correlation, 1.0)
        # Recognize identical/opposing vectors at floating-point precision;
        # their dot product can otherwise be one ulp short of +/-1.
        for i, alpha in enumerate(self._alphas):
            for j in range(i):
                for sign in (-1, 1):
                    if np.allclose(
                        alpha,
                        sign * self._alphas[j],
                        rtol=0,
                        atol=8 * np.finfo(float).eps,
                    ):
                        self._correlation[i, j] = self._correlation[j, i] = sign
        self._probabilities = norm.sf(self._betas)
        n = len(self.components)
        pairwise = np.diag(self._probabilities)
        for i in range(n):
            for j in range(i):
                ids = [i, j]
                value = self._cdf(
                    -self._betas[ids], self._correlation[np.ix_(ids, ids)]
                )
                lo = max(0.0, self._probabilities[i] + self._probabilities[j] - 1)
                hi = min(self._probabilities[i], self._probabilities[j])
                if value < lo - self.abseps or value > hi + self.abseps:
                    raise AnalysisError("Pair probability violates marginal bounds")
                pairwise[i, j] = pairwise[j, i] = np.clip(value, lo, hi)
        self._intersections = pairwise

        if isinstance(self.system, SeriesSystem):
            order = np.argsort(-self._probabilities)
            self._bounds = ditlevsen_bounds(
                self._probabilities, pairwise, ordering=order
            )
            # Disjoint events: first failed component i, all previous safe.
            # Avoid catastrophic cancellation in 1 - P(all safe).
            pf = 0.0
            for end in range(n):
                ids = order[: end + 1]
                signs = np.ones(end + 1)
                signs[-1] = -1
                covariance = self._correlation[np.ix_(ids, ids)] * np.outer(
                    signs, signs
                )
                pf += self._cdf(self._betas[ids] * signs, covariance)
        else:
            self._bounds = (
                float(
                    self._probabilities[0]
                    if n == 1
                    else max(0, self._probabilities.sum() - n + 1)
                ),
                float(self._probabilities.min()),
            )
            pf = self._cdf(-self._betas, self._correlation)
        lo, hi = self._bounds
        tolerance = self.abseps + self.releps * max(lo, abs(pf))
        if not np.isfinite(pf) or pf < lo - tolerance or pf > hi + tolerance:
            raise AnalysisError(
                "System integration is inconsistent with probability bounds"
            )
        if pf == 0 and lo > 0:
            raise AnalysisError(
                "System probability underflow; increase integration accuracy"
            )
        self._Pf = float(np.clip(pf, 0, 1))
        self._beta = float(-norm.ppf(self._Pf))
        self._results_valid = True
        return SystemFORMResult(
            method="SystemFORM",
            status="converged",
            message="Every component FORM analysis converged",
            n_limit_state_evaluations=sum(
                record.n_limit_state_evaluations for record in records.values()
            ),
            variable_names=tuple(self.model.get_variables()),
            failure_probability=self._Pf,
            beta=self._beta,
            bounds=tuple(float(bound) for bound in self._bounds),
            component_results=records,
            correlation=self._correlation,
            intersections=self._intersections,
            options=self.options,
        )
