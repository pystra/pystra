"""Importance sampling about the FORM design point."""

import numpy as np

from .form import FORM
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

    def __init__(self, analysis_options=None, limit_state=None, stochastic_model=None):
        FormAnalysis = FORM(
            stochastic_model=stochastic_model,
            limit_state=limit_state,
            analysis_options=analysis_options,
        )
        form_result = FormAnalysis.run()
        u = FormAnalysis.get_design_point()
        u = np.transpose([u])

        super().__init__(analysis_options, limit_state, stochastic_model, u)
        self._form_result = form_result

    def run(self):
        """Run importance sampling and return a :class:`SimulationResult`."""
        self.results_valid = True

        self.init_run()

        print_results = self.options.get_print_output()
        # regardless, turn off for CMC run
        self.options.set_print_output(False)
        result = CrudeMonteCarlo.run(self)
        # restore
        self.options.set_print_output(print_results)
        if self.options.get_print_output():
            self.show_results()
        return result

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
