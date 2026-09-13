#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

# import pyre library
from pystra import *

from multiprocessing import Process
import threading
import uuid
import numpy as np

# import multiprocessing as mp
import queue
import threading
import multiprocessing as mp
import subprocess
from multiprocessing import Process, Value, Array

import time
import datetime

start_time = time.time()


def example_limitstatefunction(X1, X2, X3):
    """
    example limit-state function
    """
    # Define an output queue
    output = mp.Queue()

    # function
    def LSF(j, X1, X2, X3):
        LSF_ = 1 - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2
        print("thread %s" % j)
        return output.put((j, LSF_))

    ###########################################################
    """
    speeding up through muli-threading
    """
    thread_list = []

    for i_thread in np.arange(len(X1[0, :])):
        # Instatiates the thread
        # (i) does not make a sequence, so (i,)
        t = threading.Thread(
            target=LSF,
            args=(i_thread, X1[0, i_thread], X2[0, i_thread], X3[0, i_thread]),
        )
        # Sticks the thread in a list so that it remains accessible
        thread_list.append(t)
    # Starts threads
    for thread in thread_list:
        thread.start()
    # This part is calling thread until the thread whose join() method until is terminated.
    # From http://docs.python.org/2/library/threading.html#thread-objects
    for thread in thread_list:
        thread.join()
    # sort output
    z_ = [output.get() for p in thread_list]
    z_.sort()
    z_ = [r[1] for r in z_]
    # transformation of list into array
    z = np.asanyarray(z_)
    #    print z_
    return z


############################################################################
# Define a main() function.
def main():
    # Define limit-state function  # - case 2: use predefined function
    limit_state = LimitState(example_limitstatefunction)

    # Set some options (optional)
    options = FORMOptions()
    # options.printResults(False)

    stochastic_model = StochasticModel()
    # Define random variables
    stochastic_model.add_variable(Lognormal("X1", 500, 100))
    stochastic_model.add_variable(Normal("X2", 2000, 400))
    stochastic_model.add_variable(Uniform("X3", 5, 0.5))

    # If the random variables are correlatet, then define a correlation matrix,
    # else no correlatin matrix is needed
    stochastic_model.set_correlation(
        CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
    )

    # Performe FORM analysis
    Analysis = FORM(
        options=options,
        model=stochastic_model,
        limit_state=limit_state,
    )
    result = Analysis.run()
    # Performe Distribution analysis
    # Analysis = DistributionAnalysis(options=options, model=stochastic_model, limit_state=limit_state)
    # Performe Crude Monte Carlo Simulation
    # Analysis = CrudeMonteCarlo(options=options, model=stochastic_model, limit_state=limit_state)
    #
    # Performe Importance Sampling
    # Analysis = ImportanceSampling(options=options, model=stochastic_model, limit_state=limit_state)
    #
    # Some single results:
    # beta = result.design_index
    # failure = result.failure_probability
    #    run_time = time.time() - start_time
    #    print str(datetime.timedelta(seconds=run_time))

    return result


# This is the standard boilerplate that calls the main() function.
if __name__ == "__main__":
    time_ = time.time()
    # time.t
    result = main()
    print("Done in %s seconds" % (time.time() - time_))
    print(result.design_point_x)
