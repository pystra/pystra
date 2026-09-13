#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import pystra as ra
import numpy as np


def example_limitstatefunction(r, X1, X2, X3):
    """
    example limit state function
    """
    return r - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2


# Define a main() function.
def main():
    # Define limit state function
    # - case 1: define directly as lambda function
    # limit_state = LimitState(lambda X1,X2,X3: 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2)
    # - case 2: use predefined function
    limit_state = ra.LimitState(example_limitstatefunction)

    # Set some options (optional)
    options = ra.FORMOptions()

    stochastic_model = ra.StochasticModel()
    # Define random variables
    stochastic_model.add_variable(ra.Lognormal("X1", 500, 100))
    stochastic_model.add_variable(ra.Normal("X2", 2000, 400))
    stochastic_model.add_variable(ra.Uniform("X3", 5, 0.5))

    # Define constants
    stochastic_model.add_variable(ra.Constant("r", 1.7))

    # If the random variables are correlatet, then define a correlation matrix,
    # else no correlatin matrix is needed
    stochastic_model.set_correlation(
        ra.CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
    )

    # Perform FORM analysis
    Analysis = ra.FORM(
        options=options,
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis_result = Analysis.run()
    # More detailed output
    print(Analysis_result.summary())

    # Perform SORM analysis, passing FORM result if it exists
    sorm = ra.SORM(
        options=ra.SORMOptions(),
        model=stochastic_model,
        limit_state=limit_state,
        form=Analysis,
    )
    sorm_result = sorm.run()
    # Detailed output
    print(sorm_result.summary())

    # Perform Distribution analysis
    Analysis = ra.DistributionAnalysis(
        rng=20260913,
        options=ra.SimulationOptions(),
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis_result = Analysis.run()

    # Perform Crude Monte Carlo Simulation
    Analysis = ra.CrudeMonteCarlo(
        rng=20260913,
        options=ra.SimulationOptions(),
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis_result = Analysis.run()

    # Perform Importance Sampling
    Analysis = ra.ImportanceSampling(
        rng=20260913,
        options=ra.SimulationOptions(),
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis_result = Analysis.run()

    # Some single results:
    beta = Analysis_result.beta
    pf = Analysis_result.failure_probability

    print(f"Beta is {beta}, corresponding to a failure probability of {pf}")


# This is the standard boilerplate that calls the main() function.
if __name__ == "__main__":
    main()
