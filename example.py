#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

# import pyre library
from pyre import *


import time
import datetime
start_time = time.time()


# Define a main() function.
def main():

  # Define random variables
  X1 = Lognormal('X1',500,100)
  X2 = Normal('X2',2000,400)
  X3 = Uniform('X3',5,0.5)

  # If the random variables are correlatet, then define a correlation matrix,
  # else no correlatin matrix is needed
  Corr = CorrelationMatrix([[1.0, 0.3, 0.2],
                            [0.3, 1.0, 0.2],
                            [0.2, 0.2, 1.0]])

  # Define limit state function
  # - case 1: define directly
  # g = LimitStateFunction('1 - X2*(1000*X3)**(-1) - (X1*(200*X3)**(-1))**2')
  # - case 2: define load function, wich is defined in function.py
  g = LimitStateFunction('function(X1,X2,X3)')

  # Set some options (optional)
  options = AnalysisOptions()
  # options.printResults(False)

  # Performe FORM analysis
  Analysis = Form(options)

  # Performe Distribution analysis
  # Analysis = DistributionAnalysis(options)

  # Performe Crude Monte Carlo Simulation
  # Analysis = CrudeMonteCarlo(options)

  # Performe Importance Sampling
  # Analysis = ImportanceSampling(options)

  # Some single results:
  # beta = Analysis.getBeta()
  # failure = Analysis.getFailure()

  run_time = time.time() - start_time
  print str(datetime.timedelta(seconds=run_time))


  # This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()
