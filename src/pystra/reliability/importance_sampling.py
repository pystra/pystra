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

    :Attributes:
      - analysis_option (AnalysisOption): Option for the structural analysis
      - limit_state (LimitState): Information about the limit state
      - stochastic_model (StochasticModel): Information about the model
    """

    def __init__(self, model, limit_state, *, options=None, form=None):
        super().__init__(model, limit_state, options=options)
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
                self.limitstate,
                options=FORMOptions(
                    block_size=self.options.block_size,
                    transform=self.options.transform,
                    rosenblatt_order=self.options.rosenblatt_order,
                ),
            )
            self._form_result = form.run()
            self.form = form
        elif self._form_result is None:
            if not self.form.results_valid:
                raise AnalysisError(
                    "ImportanceSampling requires a completed FORM analysis"
                )
            self._form_result = FORMResult.from_analysis(self.form)
        self.point = np.transpose([self.form.get_design_point()])
        return CrudeMonteCarlo.run(self)

    def _diagnostics(self):
        return {"form": self._form_result}

    def show_results(self):
        """Show results and plots"""
        if not self.results_valid:
            raise ValueError("Analysis not yet run")
        print("")
        print("==================================================")
        print("")
        print(" RESULTS FROM RUNNING IMPORTANCE SAMPLING")
        print("")
        print(" Reliability index beta:       ", self.beta)
        print(" Failure probability:          ", self.Pf)
        print(" Coefficient of variation of Pf", self.cov_q_bar[self.k - 1])
        print(" Number of simulations:        ", self.k)
        print("")
        print("==================================================")
        print("")
