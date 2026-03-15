#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitivity analysis of the reliability index.

This module computes the sensitivity of the FORM reliability index
with respect to the mean and standard deviation of each random
variable, using forward finite differences.
"""

from .form import Form
from .analysis import AnalysisOptions
import copy


class SensitivityAnalysis:
    r"""Numerical sensitivity analysis for the FORM reliability index.

    Computes :math:`\partial\beta / \partial\theta` for each distribution
    parameter :math:`\theta` (currently mean and standard deviation) using
    forward finite differences around the baseline FORM result.

    Parameters
    ----------
    limit_state : LimitState
        The limit state function definition.
    stochastic_model : StochasticModel
        The stochastic model (distributions + correlation).
    analysis_options : AnalysisOptions, optional
        Options forwarded to each FORM run.  Defaults are used if
        ``None``.

    Notes
    -----
    Future extensions: SORM sensitivities, correlation coefficients,
    the analytical method of [Bourinet2017]_, full distribution parameter
    coverage, and Sobol indices.

    References
    ----------
    .. [Bourinet2017] Bourinet, *FORM Sensitivities to Distribution
       Parameters with the Nataf Transformation*, in Risk and Reliability
       Analysis: Theory and Applications, Springer, 2017.
    """

    def __init__(self, limit_state, stochastic_model, analysis_options=None):
        self.limitstate = limit_state
        self.model = stochastic_model

        # Options for the calculation
        if analysis_options is None:
            self.options = AnalysisOptions()
        else:
            self.options = analysis_options

    def run_form(self, numerical=True, delta=0.01):
        r"""Run FORM-based sensitivity analysis.

        For each random variable, the mean and standard deviation are
        perturbed by ``delta * stdv`` and a new FORM analysis is
        executed.  The sensitivity is the finite-difference
        approximation
        :math:`(\beta_1 - \beta_0) / \Delta\theta`.

        Parameters
        ----------
        numerical : bool, optional
            Use numerical (finite-difference) sensitivities (default
            ``True``).  Analytical sensitivities are not yet
            implemented.
        delta : float, optional
            Relative perturbation size (default 0.01, i.e. 1 %).

        Returns
        -------
        dict
            Nested dictionary
            ``{variable_name: {"mean": d_beta_d_mean, "std": d_beta_d_std}}``.
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
        """Perturb a distribution parameter and return the actual change.

        Parameters
        ----------
        dist : Distribution
            The distribution to perturb (modified in-place).
        param : {"mean", "std"}
            Which parameter to perturb.
        delta : float
            Relative perturbation factor.

        Returns
        -------
        float
            The actual absolute change in the parameter value.
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
