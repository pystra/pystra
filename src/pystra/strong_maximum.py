"""Sphere-based diagnostic for competing FORM design regions.

The geometry follows the Strong Maximum Test described by Dutfoy and Lebrun
(2006) and documented by OpenTURNS. This implementation specializes the test
to Pystra's independent standard-normal space.
"""

from copy import copy
import math

import numpy as np
from scipy.special import betainc

from .analysis import AnalysisObject
from .form import Form
from .model import LimitState

__all__ = ["StrongMaximumTest"]


class StrongMaximumTest(AnalysisObject):
    """Search a sphere for failure points away from a candidate design point.

    Parameters
    ----------
    form : Form, optional
        Successfully converged FORM analysis. Reuses its model and transform.
        Alternatively supply all of stochastic_model, limit_state and
        design_point (in independent standard-normal coordinates).
    stochastic_model, limit_state, design_point, analysis_options : optional
        Explicit candidate inputs; cannot be combined with form. The candidate
        must be on the limit-state boundary and the origin must be safe.
    importance_level : float, default 0.15
        Density ratio epsilon in (0, 1) defining the relevant search radius.
    accuracy_level : float, default 3
        Sphere enlargement parameter tau > 1 (not an error tolerance).
    confidence_level : float, optional
        Requested cap-detection probability in (0, 1). Defaults to 0.99 if
        neither this nor point_number is supplied.
    point_number : int, optional
        Fixed number of sphere evaluations, exclusive with confidence_level.
    seed : int or numpy.random.Generator, optional
        Local random generator seed; does not use NumPy's global RNG.
    max_points : int, default 1000000
        Hard budget checked before allocating or evaluating sphere samples.
        Oversized requests raise; they are never silently reduced.

    Attributes
    ----------
    point_number, confidence_level : int, float
        Actual sample count and achieved nominal cap-detection probability.
    delta_epsilon, radius, cap_probability, vicinity_cosine : float
        Geometry and probability of hitting a reference detection cap.
    u_points, x_points : ndarray, shape (point_number, nrv)
        The same sphere sample in standard and physical coordinates.
    values : ndarray, shape (point_number,)
        Original limit-state values; failure means strictly less than zero.
    masks : dict
        Boolean masks: far_failure, near_failure, far_safe, near_safe.
    has_competing_points : bool or None
        Whether far_failure contains points; None until a successful run.
    status : str
        not_run, running, failed, competing_region_detected, or
        no_competing_region_detected. The last status is not a certificate.
    evaluation_count : int
        Evaluated sample points, including two candidate/origin checks.

    Notes
    -----
    This is a diagnostic, not a failure probability estimate or a proof of
    global optimality. Nominal confidence concerns hitting a fixed spherical
    cap under the test's local-plane/extent assumptions, not the probability
    that FORM is correct. Small bounded failure islands inside the sphere can
    be missed entirely. Far failure points are restart candidates, not new
    design points. No gradient-based interpretation of their g magnitude is
    made. The test rejects origin-in-failure and near-zero-radius candidates.

    References
    ----------
    Dutfoy, A. and Lebrun, R. (2006). The Strong Maximum Test: an efficient way
    to assess the quality of a design point. PSAM8, New Orleans.
    OpenTURNS theory: https://openturns.github.io/openturns/latest/theory/
    reliability_sensitivity/strong_maximum_test.html
    """

    def __init__(
        self,
        form=None,
        *,
        stochastic_model=None,
        limit_state=None,
        design_point=None,
        analysis_options=None,
        importance_level=0.15,
        accuracy_level=3.0,
        confidence_level=None,
        point_number=None,
        seed=None,
        max_points=1000000,
    ):
        if form is not None:
            if (
                not isinstance(form, Form)
                or not form.results_valid
                or not form.converged
            ):
                raise ValueError("Supply a successfully converged Form analysis")
            if any(
                arg is not None
                for arg in (
                    stochastic_model,
                    limit_state,
                    design_point,
                    analysis_options,
                )
            ):
                raise ValueError(
                    "form cannot be combined with explicit candidate inputs"
                )
            stochastic_model, limit_state = form.model, form.limitstate
            analysis_options = form.options
            design_point = form.getDesignPoint()
            if form.getBeta() <= 0:
                raise ValueError("Strong Maximum Test requires a safe-origin candidate")
            if form.transform.standard_space != "normal":
                raise ValueError(
                    "Strong Maximum Test currently requires independent normal space; select Rosenblatt"
                )
        elif any(arg is None for arg in (stochastic_model, limit_state, design_point)):
            raise ValueError("Supply form or model, limit_state and design_point")
        # Independent evaluator state; do not overwrite the Form limit state's
        # most recent x/gradient evaluation when running a diagnostic.
        super().__init__(
            stochastic_model=stochastic_model,
            limit_state=LimitState(limit_state.expression),
            analysis_options=copy(analysis_options),
        )
        self.form = form
        if form is not None:
            self.transform = copy(form.transform)
        self.design_point = np.asarray(design_point, dtype=float).copy()
        self.nrv = self.model.getLenMarginalDistributions()
        if self.design_point.shape != (self.nrv,) or not np.all(
            np.isfinite(self.design_point)
        ):
            raise ValueError("design_point must be a finite vector in full model order")
        self.beta = float(np.linalg.norm(self.design_point))
        if not np.isfinite(self.beta) or self.beta <= np.sqrt(np.finfo(float).eps):
            raise ValueError(
                "Design point is too close to the origin or has invalid norm"
            )
        if not np.isfinite(importance_level) or not 0 < importance_level < 1:
            raise ValueError("importance_level must be finite and in (0, 1)")
        if not np.isfinite(accuracy_level) or accuracy_level <= 1:
            raise ValueError("accuracy_level must be finite and greater than 1")
        self.importance_level, self.accuracy_level = importance_level, accuracy_level
        self.max_points = self._positive_integer(max_points, "max_points")
        if confidence_level is not None and point_number is not None:
            raise ValueError("Specify confidence_level or point_number, not both")
        if confidence_level is None and point_number is None:
            confidence_level = 0.99
        self.requested_confidence_level = confidence_level
        if confidence_level is not None and (
            not np.isfinite(confidence_level) or not 0 < confidence_level < 1
        ):
            raise ValueError("confidence_level must be finite and in (0, 1)")

        relevant_radius = np.hypot(self.beta, np.sqrt(-2 * np.log(importance_level)))
        increment = (-2 * np.log(importance_level)) / (relevant_radius + self.beta)
        self.delta_epsilon = increment / self.beta
        self.radius = self.beta + accuracy_level * increment
        self.vicinity_cosine = self.beta / self.radius
        cosine = relevant_radius / self.radius
        sin_squared = (1 - cosine) * (1 + cosine)
        self.cap_probability = float(
            0.5
            if self.nrv == 1
            else 0.5 * betainc((self.nrv - 1) / 2, 0.5, sin_squared)
        )
        if not np.isfinite(self.radius) or not 0 < self.cap_probability <= 0.5:
            raise ValueError(
                "Detection cap is numerically unresolved; change test parameters"
            )
        log_miss = np.log1p(-self.cap_probability)
        if point_number is None:
            point_number = math.ceil(np.log1p(-confidence_level) / log_miss)
            if -np.expm1(point_number * log_miss) < confidence_level:
                point_number += 1
        self.point_number = self._positive_integer(point_number, "point_number")
        if self.point_number > self.max_points:
            raise ValueError(
                f"Test needs {self.point_number} sphere points, exceeding "
                f"max_points={self.max_points}; adjust the budget or parameters"
            )
        self.confidence_level = float(-np.expm1(self.point_number * log_miss))
        self.rng = np.random.default_rng(seed)
        self._clear_results()

    @staticmethod
    def _positive_integer(value, name):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _clear_results(self):
        self.results_valid = False
        self.status = "not_run"
        self.has_competing_points = None
        self.u_points = self.x_points = self.values = None
        self.masks = {}
        self.evaluation_count = 0

    def _evaluate(self, u):
        marg = self.model.getMarginalDistributions()
        x = np.array([self.transform.u_to_x(point, marg) for point in u])
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite physical coordinates on test sphere")
        values, _ = self.limitstate.evaluate_lsf(x.T, self.model, self.options, "no")
        self.evaluation_count += len(u)
        values = np.asarray(values).reshape(-1)
        if values.shape != (len(u),) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Nonfinite or invalid limit-state values in Strong Maximum Test"
            )
        return x, values

    def run(self):
        """Evaluate the sphere and classify points by failure and vicinity."""
        self._clear_results()
        self.status = "running"
        try:
            if self.form is None:
                self.init_run()
            block = self._positive_integer(self.options.getBlockSize(), "block_size")
            _, checks = self._evaluate(
                np.array([np.zeros(self.nrv), self.design_point])
            )
            if checks[0] <= 0:
                raise ValueError(
                    "Strong Maximum Test requires the origin to be strictly safe"
                )
            if abs(checks[1]) > self.options.getE1() * abs(checks[0]):
                raise ValueError(
                    "Candidate design point is not on the limit-state boundary"
                )
            u = np.empty((self.point_number, self.nrv))
            x = np.empty_like(u)
            values = np.empty(self.point_number)
            for start in range(0, self.point_number, block):
                stop = min(self.point_number, start + block)
                directions = self.rng.standard_normal((stop - start, self.nrv))
                lengths = np.linalg.norm(directions, axis=1)
                while np.any(lengths == 0):
                    zero = lengths == 0
                    directions[zero] = self.rng.standard_normal(
                        (np.count_nonzero(zero), self.nrv)
                    )
                    lengths = np.linalg.norm(directions, axis=1)
                u[start:stop] = self.radius * directions / lengths[:, None]
                x[start:stop], values[start:stop] = self._evaluate(u[start:stop])
            near = u @ (self.design_point / self.beta) > self.beta
            failure = values < 0
            self.u_points, self.x_points, self.values = u, x, values
            self.masks = dict(
                far_failure=~near & failure,
                near_failure=near & failure,
                far_safe=~near & ~failure,
                near_safe=near & ~failure,
            )
            self.has_competing_points = bool(np.any(self.masks["far_failure"]))
            self.status = (
                "competing_region_detected"
                if self.has_competing_points
                else "no_competing_region_detected"
            )
            self.results_valid = True
        except Exception:
            self.status = "failed"
            raise
        if self.options.getPrintOutput():
            self.showResults()

    def getPoints(self, region="far_failure", uspace=True):
        """Return selected points as rows; default is far failure in U-space."""
        if not self.results_valid:
            raise ValueError("Strong Maximum Test has no valid result")
        return (self.u_points if uspace else self.x_points)[self.masks[region]]

    def getValues(self, region="far_failure"):
        """Return original limit-state values for the selected point group."""
        if not self.results_valid:
            raise ValueError("Strong Maximum Test has no valid result")
        return self.values[self.masks[region]]

    def showResults(self):
        """Print the diagnostic outcome and its nominal sampling confidence."""
        self.getPoints()
        print(f"Strong Maximum Test: {self.status}")
        print(
            f"Sphere points: {self.point_number}; nominal confidence: {self.confidence_level:.6g}"
        )
        print({name: int(mask.sum()) for name, mask in self.masks.items()})
