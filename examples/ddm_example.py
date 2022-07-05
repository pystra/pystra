# -*- coding: utf-8 -*-

import pystra as ra
import numpy as np
import timeit


def lsf(r, X1, X2, X3, X4, X5, X6):
    """
    Calrel example from FERUM
    """
    G = (
        r
        - X2 / (1000 * X3)
        - (X1 / (200 * X3)) ** 2
        - X5 / (1000 * X6)
        - (X4 / (200 * X6)) ** 2
    )
    grad_G = np.array(
        [
            -X1 / (20000 * X3**2),
            -1 / (1000 * X3),
            (20 * X2 * X3 + X1**2) / (20000 * X3**3),
            -X4 / (20000 * X6**2),
            -1 / (1000 * X6),
            (20 * X5 * X6 + X4**2) / (20000 * X6**3),
        ]
    )
    return G, grad_G


def run(diff_mode):
    limit_state = ra.LimitState(lsf)

    # Set some options (optional)
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)
    options.setDiffMode(diff_mode)

    stochastic_model = ra.StochasticModel()

    # Define random variables
    stochastic_model.addVariable(ra.Lognormal("X1", 500, 100))
    stochastic_model.addVariable(ra.Lognormal("X2", 2000, 400))
    stochastic_model.addVariable(ra.Uniform("X3", 5, 0.5))
    stochastic_model.addVariable(ra.Lognormal("X4", 450, 90))
    stochastic_model.addVariable(ra.Lognormal("X5", 1800, 360))
    stochastic_model.addVariable(ra.Uniform("X6", 4.5, 0.45))

    # Define constants
    stochastic_model.addVariable(ra.Constant("r", 1.7))

    stochastic_model.setCorrelation(
        ra.CorrelationMatrix(
            [
                [1.0, 0.3, 0.2, 0, 0, 0],
                [0.3, 1.0, 0.2, 0, 0, 0],
                [0.2, 0.2, 1.0, 0, 0, 0],
                [0, 0, 0, 1.0, 0.3, 0.2],
                [0, 0, 0, 0.3, 1.0, 0.2],
                [0, 0, 0, 0.2, 0.2, 1.0],
            ]
        )
    )

    # Set up FORM analysis
    form = ra.Form(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    # Run it
    form.run()


def run_ffd():
    run("ffd")


def run_ddm():
    run("ddm")


# Define a main() function.
def main():
    number = 100
    time_ffd = timeit.timeit(stmt=run_ffd, number=number)
    time_ddm = timeit.timeit(stmt=run_ddm, number=number)

    print("Total time taken (s)")
    print(f"FFD: {time_ffd}; DDM: {time_ddm}")
    print("Average time per call (s):")
    print(f"FFD: {time_ffd/number}; DDM: {time_ddm/number}")
    print(f"DDM speed-up: {time_ffd/time_ddm:.2f}")


if __name__ == "__main__":
    main()
