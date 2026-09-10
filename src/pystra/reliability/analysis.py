"""Base class for reliability analyses.

:class:`AnalysisObject` is the common base of the FORM, SORM, simulation and
diagnostic analyses. Their settings are the frozen objects in
:mod:`pystra.options`.
"""

import numpy as np

from ..errors import ModelError
from ..model import StochasticModel, LimitState
from ..dependence.transformation import Transformation
from ..dependence.correlation import set_modified_correlation_matrix

__all__ = ["AnalysisObject"]


def _check_on_failure(on_failure):
    """Reject an unknown failure policy."""
    if on_failure not in ("raise", "return"):
        raise ModelError("on_failure must be 'raise' or 'return'")
    return on_failure


def _check_rng(rng):
    """Reject an ``rng`` that cannot seed a NumPy generator."""
    if not isinstance(rng, np.random.Generator):
        np.random.default_rng(rng)
    return rng


def _generator(rng):
    """Return the random generator for one run.

    A ``numpy.random.Generator`` is used as given, so it advances its own
    state; a seed or None makes a new generator, so an integer seed recreates
    the same stream on every run.
    """
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


class AnalysisObject:
    """Base class for reliability analyses.

    Stores the stochastic model, limit state and settings, and provides
    ``init_run``, which computes the Nataf correlation and the
    isoprobabilistic transformation before each run.

    Parameters
    ----------
    model : StochasticModel
        The probabilistic model.
    limit_state : LimitState
        The limit state function.
    options : optional
        The analysis's settings, an instance of its ``_options_type``;
        the defaults if omitted.

    Attributes
    ----------
    model : StochasticModel
    limitstate : LimitState
    options
        The frozen settings.
    transform : Transformation
    results_valid : bool
        ``True`` after a successful ``run()``.
    """

    _options_type = None
    _requires_limit_state = True

    def __init__(self, model, limit_state, options=None):
        name = type(self).__name__
        if not isinstance(model, StochasticModel):
            raise TypeError(
                f"{name} requires a StochasticModel, not {type(model).__name__}"
            )
        if not isinstance(limit_state, LimitState) and (
            limit_state is not None or self._requires_limit_state
        ):
            raise TypeError(
                f"{name} requires a LimitState, not {type(limit_state).__name__}"
            )
        if options is None:
            options = self._options_type()
        elif not isinstance(options, self._options_type):
            raise TypeError(
                f"{name} takes {self._options_type.__name__}, not {type(options).__name__}"
            )
        self.model = model
        self.limit_state = limit_state
        self.options = options

        transform, _ = self._dependence()
        self.transform = Transformation(
            transform_type=None if transform in ("nataf", "rosenblatt") else transform
        )

        self._results_valid = False
        self._n_evaluations = 0

    def _dependence(self):
        """Return the ``(transform, rosenblatt_order)`` settings for this run."""
        return self.options.transform, self.options.rosenblatt_order

    def _settings(self):
        """Return the settings that govern limit-state evaluation."""
        return self.options

    def _count(self, n):
        """Record limit-state function calls made by this analysis."""
        self._n_evaluations += n

    def _lsf(self, x, gradient=False):
        """Evaluate the limit state at the columns of *x*, counting the calls.

        With *gradient*, the gradient is computed with the configured
        differentiation; otherwise only values are computed.
        """
        settings = self._settings()
        kwargs = dict(block_size=settings.block_size, counter=self._count)
        if gradient:
            kwargs.update(
                differentiation=settings.differentiation,
                ffd_parameter=settings.ffd_parameter,
            )
        return self.limit_state.evaluate_lsf(x, self.model, **kwargs)

    def init_run(self):
        """Initialise the model's isoprobabilistic transformation.

        Uses the explicit copula or calibrates the legacy Gaussian Nataf
        correlation. Must be called at the start of every
        ``run()`` method in subclasses.
        """

        self._n_evaluations = 0
        copula = self.model.get_copula()
        selected, order = self._dependence()
        if copula is not None or selected in ("nataf", "rosenblatt"):
            from ..dependence.copula import GaussianCopula, StudentTCopula
            from ..dependence.joint import CopulaTransformation

            joint = self.model.get_joint_distribution()
            gaussian = isinstance(joint.copula, GaussianCopula) and not isinstance(
                joint.copula, StudentTCopula
            )
            method = selected
            if selected is None:
                method = "nataf" if gaussian else "rosenblatt"
            if selected in ("cholesky", "svd"):
                method = "nataf"
            self.transform = CopulaTransformation(
                joint,
                method=method,
                order=order,
                factorization="svd" if selected == "svd" else "cholesky",
            )
            if self.transform.standard_space != "normal" and not getattr(
                self, "supports_spherical_space", False
            ):
                self._results_valid = False
                raise ValueError(
                    "This analysis requires independent normal space; select Rosenblatt for this copula"
                )
            if gaussian:
                self.model.set_modified_correlation(joint.copula.correlation)
            return

        if order is not None:
            raise ValueError(
                "Conditioning order requires the Rosenblatt transformation"
            )
        self.transform = Transformation(selected)

        # Computation of modified correlation matrix R0
        set_modified_correlation_matrix(self.model)

        # Compute the isoprobabilistic transform
        self.transform.compute(self.model.get_modified_correlation())
