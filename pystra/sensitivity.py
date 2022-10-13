#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .form import Form
from .analysis import AnalysisOptions
import copy


class SensitivityAnalysis:
    """
    Implements sensitivity analyses for the reliability index.

    Current implementation is numerical and for FORM only, and only covers the
    first and second moments.

    Future extensions are to extend to SORM; to include correlation
    coefficients; to add the analytical method of Bourinet (2017);
    to cover all distribution parameters, and include Sobol indices.

    Bourinet (2017), FORM Sensitivities to Distribution Parameters with the
    Nataf Transformation, P. Gardoni (ed.), Risk and Reliability Analysis:
    Theory and Applications, Springer Series in Reliability Engineering,
    DOI 10.1007/978-3-319-52425-2_12

    """

    def __init__(self, limit_state, stochastic_model, analysis_options=None):
        """
        Store the problem definition
        """

        self.limitstate = limit_state
        self.model = stochastic_model

        # Options for the calculation
        if analysis_options is None:
            self.options = AnalysisOptions()
        else:
            self.options = analysis_options

    def run_form(self, numerical=True, delta=0.01):
        """
        numerical = True (default)
        delta = the relative change in parameter value
        """

        if numerical is False:
            print(
                "Analytical sensitivity analysis is not yet implemented:"
                "defaulting to numerical sensitivity analysis"
            )

        variables = self.model.getVariables()
        names = variables.keys()

        sensitivities = {n: {"mean": 0, "std": 0} for n in names}

        # Get the base result
        form = Form(stochastic_model=self.model, limit_state=self.limitstate)
        form.run()
        beta0 = form.getBeta()

        for param in ["mean", "std"]:
            for name in names:
                model1 = copy.deepcopy(self.model)
                dist = model1.getVariable(name)

                delta_actual = self._change_param(dist, param, delta)

                # Calculate and store the sensitivity
                form = Form(stochastic_model=model1, limit_state=self.limitstate)
                form.run()
                beta1 = form.getBeta()
                sens = (beta1 - beta0) / delta_actual
                sensitivities[name][param] = sens

        return sensitivities

    def _change_param(self, dist, param, delta):
        """
        Finds the right parameter, adjusts it, and returns new distribution
        object
        """

        if param == "mean":
            p0 = dist.mean
            dist.set_location(p0 + delta * dist.stdv)
            p1 = dist.mean
        else:
            p0 = dist.stdv
            dist.set_scale(p0 + delta * dist.stdv)
            p1 = dist.stdv

        return p1 - p0
