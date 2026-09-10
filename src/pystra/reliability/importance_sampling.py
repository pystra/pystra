"""Importance sampling about the FORM design point."""

import numpy as np

from .form import FORM
from ..errors import AnalysisError
from ..options import FORMOptions
from ..results import FORMResult
from .monte_carlo import CrudeMonteCarlo

__all__ = ["ImportanceSampling"]


class ImportanceSampling(CrudeMonteCarlo):
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
        A completed FORM analysis whose design point centres the samples. If
        None, :meth:`run` first runs FORM with this analysis's block size and
        transformation.
    """

    def __init__(self, model, limit_state, *, options=None, form=None, rng=None):
        super().__init__(model, limit_state, options=options, rng=rng)
        if form is not None and not isinstance(form, FORM):
            raise TypeError("form must be a FORM analysis")
        self.form = form
        self._form_result = None

    def run(self):
        """Run importance sampling and return a :class:`SimulationResult`.

        Samples are centred on the FORM design point. If no completed FORM
        analysis was supplied, FORM is run first with this analysis's block
        size and transformation.
        """
        if self.form is None:
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
            self.form = form
        elif self._form_result is None:
            if not self.form._results_valid:
                raise AnalysisError(
                    "ImportanceSampling requires a completed FORM analysis"
                )
            self._form_result = FORMResult.from_analysis(self.form)
        self.point = np.transpose([self.form._u])
        return CrudeMonteCarlo.run(self)

    def _diagnostics(self):
        return {**super()._diagnostics(), "form": self._form_result}
