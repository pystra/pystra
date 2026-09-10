"""Sphere-based diagnostic for competing FORM design regions.

The geometry follows the Strong Maximum Test described by Dutfoy and Lebrun
(2006) and documented by OpenTURNS. This implementation specializes the test
to Pystra's independent standard-normal space.
"""

from copy import copy
import math

import numpy as np
from scipy.special import betainc

from .analysis import AnalysisObject, _check_rng, _generator
from .form import FORM
from ..model import LimitState
from ..options import FORMOptions
from ..results import StrongMaximumResult

__all__ = ["StrongMaximumTest"]


class StrongMaximumTest(AnalysisObject):
    """Search a sphere for failure points away from a candidate design point.

    Parameters
    ----------
    form : FORM, optional
        Successfully converged FORM analysis. Reuses its model, transform and
        settings. Alternatively supply all of model, limit_state and
        design_point (in independent standard-normal coordinates).
    model, limit_state, design_point, options : optional
        Explicit candidate inputs and FORMOptions; cannot be combined with
        form. The candidate must be on the limit-state boundary and the
        origin must be safe.
    importance_level : float, default 0.15
        Density ratio epsilon in (0, 1) defining the relevant search radius.
    accuracy_level : float, default 3
        Sphere enlargement parameter tau > 1 (not an error tolerance).
    confidence_level : float, optional
        Requested cap-detection probability in (0, 1). Defaults to 0.99 if
        neither this nor point_number is supplied.
    point_number : int, optional
        Fixed number of sphere evaluations, exclusive with confidence_level.
    rng : int, numpy.random.Generator or None, optional
        Random source; NumPy's global generator is not used. A seed recreates
        the same sphere sample on every run; a generator advances its own
        state.
    max_points : int, default 1000000
        Hard budget checked before allocating or evaluating sphere samples.
        Oversized requests raise; they are never silently reduced.

    Attributes
    ----------
    point_number, confidence_level : int, float
        Actual sample count and achieved nominal cap-detection probability.
    delta_epsilon, radius, cap_probability, vicinity_cosine : float
        Geometry and probability of hitting a reference detection cap.

    :meth:`run` returns a :class:`~pystra.results.StrongMaximumResult` with the
    sphere points, their limit-state values and region masks, and whether a
    competing region was found. Finding none is not a certificate.

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

    _options_type = FORMOptions

    def __init__(
        self,
        form=None,
        *,
        model=None,
        limit_state=None,
        design_point=None,
        options=None,
        importance_level=0.15,
        accuracy_level=3.0,
        confidence_level=None,
        point_number=None,
        rng=None,
        max_points=1000000,
    ):
        if form is not None:
            if (
                not isinstance(form, FORM)
                or not form._results_valid
                or not form._converged
            ):
                raise ValueError("Supply a successfully converged FORM analysis")
            if any(
                arg is not None
                for arg in (
                    model,
                    limit_state,
                    design_point,
                    options,
                )
            ):
                raise ValueError(
                    "form cannot be combined with explicit candidate inputs"
                )
            model, limit_state = form.model, form.limit_state
            options = form.options
            design_point = form._u
            if form._beta <= 0:
                raise ValueError("Strong Maximum Test requires a safe-origin candidate")
            if form.transform.standard_space != "normal":
                raise ValueError(
                    "Strong Maximum Test currently requires independent normal space; select Rosenblatt"
                )
        elif any(arg is None for arg in (model, limit_state, design_point)):
            raise ValueError("Supply form or model, limit_state and design_point")
        # Independent evaluator state; do not overwrite the FORM limit state's
        # most recent x/gradient evaluation when running a diagnostic.
        super().__init__(model, LimitState(limit_state.expression), options)
        self.form = form
        if form is not None:
            self.transform = copy(form.transform)
        self.design_point = np.asarray(design_point, dtype=float).copy()
        self._nrv = self.model.get_len_marginal_distributions()
        if self.design_point.shape != (self._nrv,) or not np.all(
            np.isfinite(self.design_point)
        ):
            raise ValueError("design_point must be a finite vector in full model order")
        self._beta = float(np.linalg.norm(self.design_point))
        if not np.isfinite(self._beta) or self._beta <= np.sqrt(np.finfo(float).eps):
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

        relevant_radius = np.hypot(self._beta, np.sqrt(-2 * np.log(importance_level)))
        increment = (-2 * np.log(importance_level)) / (relevant_radius + self._beta)
        self.delta_epsilon = increment / self._beta
        self.radius = self._beta + accuracy_level * increment
        self.vicinity_cosine = self._beta / self.radius
        cosine = relevant_radius / self.radius
        sin_squared = (1 - cosine) * (1 + cosine)
        self.cap_probability = float(
            0.5
            if self._nrv == 1
            else 0.5 * betainc((self._nrv - 1) / 2, 0.5, sin_squared)
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
        self.rng = _check_rng(rng)
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
        self._results_valid = False
        self._status = "not_run"
        self._has_competing_points = None
        self._u_points = self._x_points = self._values = None
        self._masks = {}
        self._evaluation_count = 0

    def _evaluate(self, u):
        marg = self.model.get_marginal_distributions()
        x = np.array([self.transform.u_to_x(point, marg) for point in u])
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite physical coordinates on test sphere")
        values, _ = self._lsf(x.T)
        self._evaluation_count += len(u)
        values = np.asarray(values).reshape(-1)
        if values.shape != (len(u),) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Nonfinite or invalid limit-state values in Strong Maximum Test"
            )
        return x, values

    def run(self):
        """Evaluate the sphere and classify points by failure and vicinity.

        Returns a :class:`StrongMaximumResult`.
        """
        self._clear_results()
        self._status = "running"
        random = _generator(self.rng)
        try:
            if self.form is None:
                self.init_run()
            block = self._positive_integer(self.options.block_size, "block_size")
            _, checks = self._evaluate(
                np.array([np.zeros(self._nrv), self.design_point])
            )
            if checks[0] <= 0:
                raise ValueError(
                    "Strong Maximum Test requires the origin to be strictly safe"
                )
            if abs(checks[1]) > self.options.limit_state_tolerance * abs(checks[0]):
                raise ValueError(
                    "Candidate design point is not on the limit-state boundary"
                )
            u = np.empty((self.point_number, self._nrv))
            x = np.empty_like(u)
            values = np.empty(self.point_number)
            for start in range(0, self.point_number, block):
                stop = min(self.point_number, start + block)
                directions = random.standard_normal((stop - start, self._nrv))
                lengths = np.linalg.norm(directions, axis=1)
                while np.any(lengths == 0):
                    zero = lengths == 0
                    directions[zero] = random.standard_normal(
                        (np.count_nonzero(zero), self._nrv)
                    )
                    lengths = np.linalg.norm(directions, axis=1)
                u[start:stop] = self.radius * directions / lengths[:, None]
                x[start:stop], values[start:stop] = self._evaluate(u[start:stop])
            near = u @ (self.design_point / self._beta) > self._beta
            failure = values < 0
            self._u_points, self._x_points, self._values = u, x, values
            self._masks = dict(
                far_failure=~near & failure,
                near_failure=near & failure,
                far_safe=~near & ~failure,
                near_safe=near & ~failure,
            )
            self._has_competing_points = bool(np.any(self._masks["far_failure"]))
            self._status = (
                "competing_region_detected"
                if self._has_competing_points
                else "no_competing_region_detected"
            )
            self._results_valid = True
        except Exception:
            self._status = "failed"
            raise
        return StrongMaximumResult(
            method="StrongMaximumTest",
            status="completed",
            message=(
                "Competing failure region detected"
                if self._has_competing_points
                else "No competing failure region detected; this is not a certificate"
            ),
            n_limit_state_evaluations=self._evaluation_count,
            variable_names=tuple(self.model.get_variables()),
            has_competing_points=self._has_competing_points,
            design_point_u=self.design_point,
            design_index=self._beta,
            radius=self.radius,
            point_number=self.point_number,
            confidence_level=self.confidence_level,
            cap_probability=self.cap_probability,
            points_u=self._u_points,
            points_x=self._x_points,
            limit_state_values=self._values,
            regions=self._masks,
            options=self.options,
        )
