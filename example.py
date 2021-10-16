#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import pystra as pr
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
    limit_state = pr.LimitState(example_limitstatefunction)

    # Set some options (optional)
    options = pr.AnalysisOptions()
    # options.printResults(False)

    stochastic_model = pr.StochasticModel()
    # Define random variables
    stochastic_model.addVariable(pr.Lognormal("X1", 500, 100))
    stochastic_model.addVariable(pr.Normal("X2", 2000, 400))
    stochastic_model.addVariable(pr.Uniform("X3", 5, 0.5))

    # Define constants
    stochastic_model.addVariable(pr.Constant("r", 1))

    # If the random variables are correlatet, then define a correlation matrix,
    # else no correlatin matrix is needed
    stochastic_model.setCorrelation(
        pr.CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
    )

    # Perform FORM analysis
    Analysis = pr.Form(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    # More detailed output
    Analysis.showDetailedOutput()

    # Perform SORM analysis, passing FORM result if it exists
    sorm = pr.Sorm(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
        form=Analysis,
    )
    sorm.run()
    # Detailed output
    sorm.showDetailedOutput()

    # Perform Distribution analysis
    Analysis = pr.DistributionAnalysis(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )

    # Perform Crude Monte Carlo Simulation
    Analysis = pr.CrudeMonteCarlo(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )

    # Perform Importance Sampling
    Analysis = pr.ImportanceSampling(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )

    # Some single results:
    beta = Analysis.getBeta()
    failure = Analysis.getFailure()

    print(
        "Beta is {}, corresponding to a failure probability of {}".format(beta, failure)
    )
    

# This is the standard boilerplate that calls the main() function.
if __name__ == "__main__":
    main()

