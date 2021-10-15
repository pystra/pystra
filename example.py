#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

# import pyre library
from pystra import *

from scipy.stats import norm, lognorm, uniform
import numpy as np

import time
import datetime

start_time = time.time()


def example_limitstatefunction(g, X1, X2, X3):
    """
    example limit state function
    """
    return g - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2


# Define a main() function.
def main():

    # Define limit state function
    # - case 1: define directly as lambda function
    # limit_state = LimitState(lambda X1,X2,X3: 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2)
    # - case 2: use predefined function
    limit_state = LimitState(example_limitstatefunction)

    # Set some options (optional)
    options = AnalysisOptions()
    # options.printResults(False)

    stochastic_model = StochasticModel()
    # Define random variables
    #stochastic_model.addVariable(Lognormal("X1", 500, 100))
    #stochastic_model.addVariable(Normal("X2", 2000, 400))
    #stochastic_model.addVariable(Uniform("X3", 5, 0.5))
    
    # Define random variables (using Scipy)
    ## Lognormal
    zeta = (np.log(1+(100/500)**2))**0.5
    lamb = np.log(500) - 0.5*zeta**2
    stochastic_model.addVariable(ScipyDist("X1", lognorm(s=zeta,
                                                         scale=np.exp(lamb))))
    # Normal
    stochastic_model.addVariable(ScipyDist("X2", norm(loc=2000, scale=400)))
    ## Uniform
    a_b = (0.5**2*12)**(1/2)
    a = (2*5-a_b)/2
    stochastic_model.addVariable(ScipyDist("X3", uniform(loc=a,scale=a_b)))
    
    # Define constants
    stochastic_model.addVariable(Constant("g", 1))

    # If the random variables are correlatet, then define a correlation matrix,
    # else no correlatin matrix is needed
    stochastic_model.setCorrelation(
        CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
    )

    # Perform FORM analysis
    Analysis = Form(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    # More detailed output
    Analysis.showDetailedOutput()

    # Perform SORM analysis, passing FORM result if it exists
    sorm = Sorm(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
        form=Analysis,
    )
    sorm.run()
    # Detailed output
    sorm.showDetailedOutput()

    # Perform Distribution analysis
    Analysis = DistributionAnalysis(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )

    # Perform Crude Monte Carlo Simulation
    Analysis = CrudeMonteCarlo(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )

    # Perform Importance Sampling
    Analysis = ImportanceSampling(
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
    run_time = time.time() - start_time
    print(str(datetime.timedelta(seconds=run_time)))

    # This is the standard boilerplate that calls the main() function.


if __name__ == "__main__":
    main()
