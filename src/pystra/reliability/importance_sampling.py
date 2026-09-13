"""Importance sampling about the FORM design point."""

import numpy as np

import pystra as _pystra
from ..dependence.copula import _RandomSeed
from .form import FORM
from ._form_reuse import _check_form, _check_coordinates, _FORMReuse
from ..options import FORMOptions
from ..results import FORMResult
from .monte_carlo import CrudeMonteCarlo

__all__ = ["ImportanceSampling"]


class ImportanceSampling(_FORMReuse, CrudeMonteCarlo):
    """Importance Sampling

    To decrease the number of simulations and the coefficient of variation,
    other methods can be performed. One commonly applied method is the
    Importance Sampling simulation method (IS).

    Parameters
    ----------
    model : StochasticModel
    limit_state : LimitState
    options : SimulationOptions, optional
    rng : int, numpy.random.Generator or None, optional
        Random source; NumPy's global generator is not used. A seed recreates
        the same stream on every run, a generator advances its own state, and
        None draws fresh entropy.
    form : FORM, optional
        A completed FORM analysis whose design point centers the samples. If
        None, :meth:`run` first runs FORM with this analysis's block size and
        transformation.
    """

    def __init__(
        self,
        model: "_pystra.StochasticModel",
        limit_state: "_pystra.LimitState",
        *,
        options: "_pystra.SimulationOptions | None" = None,
        form: "_pystra.FORM | None" = None,
        rng: _RandomSeed = None,
    ) -> None:
        super().__init__(model, limit_state, options=options, rng=rng)
        if form is not None and not isinstance(form, FORM):
            raise TypeError("form must be a FORM analysis")
        self.form = form
        self._form_result = None

    def run(self) -> "_pystra.SimulationResult":
        """Run importance sampling and return a :class:`SimulationResult`.

        Samples are centered on the FORM design point. If no completed FORM
        analysis was supplied, FORM is run first with this analysis's block
        size and transformation.
        """
        self._results_valid = False
        self._Pf = self._beta = None
        if self._supplied_form is None:
            form = FORM(
                self.model,
                self.limit_state,
                on_failure="return",
                options=FORMOptions(
                    block_size=self.options.block_size,
                    transform=self.options.transform,
                    rosenblatt_order=self.options.rosenblatt_order,
                ),
            )
            self._form_result = form.run()
            self._form = form
        _check_form(self.form, self.model, self.limit_state)
        self._form_result = FORMResult.from_analysis(self.form)
        self.point = np.transpose([self.form._u])
        return CrudeMonteCarlo.run(self)

    def _set_point(self, point=None):
        # Monte Carlo prepares its transform before setting the sample center.
        _check_coordinates(self.form, self.transform)
        super()._set_point(point)

    def _diagnostics(self):
        return {**super()._diagnostics(), "form": self._form_result}
