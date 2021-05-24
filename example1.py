#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

# import pyre library
from pyre import *


import time
import datetime
start_time = time.time()


def example_limitstatefunction(m1, m2, p, gamma):
    """
    example limit state function
    """
    return 1 - (m1/(0.030*gamma))-(m2/(0.015*gamma))-(p/(0.190*gamma))**(2)
    #return 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2


# Define a main() function.
def main():

    # Define limit state function
    # - case 1: define directly as lambda function
    #limit_state = LimitState(lambda X1,X2,X3: 1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2)
    # - case 2: use predefined function
    limit_state = LimitState(example_limitstatefunction)

    # Set some options (optional)
    options = AnalysisOptions()
    # options.printResults(False)

    stochastic_model = StochasticModel()
    # Define random variables
    stochastic_model.addVariable(Normal('m1', 250, 75))
    stochastic_model.addVariable(Normal('m2', 125, 37.5))
    stochastic_model.addVariable(Gumbel('p', 2500, 500))
    stochastic_model.addVariable(Weibull('gamma', 40, 4))
    

    # If the random variables are correlatet, then define a correlation matrix,
    # else no correlatin matrix is needed
    stochastic_model.setCorrelation(CorrelationMatrix([[1.0, 0, 0, 0],
                                                       [0.5, 1.0, 0, 0],
                                                       [0.3, 0.3, 1.0, 0],
                                                       [0, 0, 0, 1.0]]))

    # Perform FORM analysis
    Analysis = Form(analysis_options=options,
                    stochastic_model=stochastic_model, limit_state=limit_state)
    # More detailed output
    Analysis.showDetailedOutput()

    # Perform SORM analysis, passing FORM result if it exists
    sorm = Sorm(analysis_options=options,stochastic_model=stochastic_model, 
                limit_state=limit_state, form=Analysis)
    sorm.run(fit_type='pf')
    # Detailed output
    sorm.showDetailedOutput()


    # Some single results:
    beta = Analysis.getBeta()
    failure = Analysis.getFailure()

    print("Beta is {}, corresponding to a failure probability of {}".format(beta, failure))
    run_time = time.time() - start_time
    print(str(datetime.timedelta(seconds=run_time)))

    # This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
