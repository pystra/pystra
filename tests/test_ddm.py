import pytest
import pystra as ra
import numpy as np


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


def setup(diff_mode):
    limit_state = ra.LimitState(lsf)

    options = ra.AnalysisOptions()
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

    return options, stochastic_model, limit_state


def test_ddm_form():

    options, stochastic_model, limit_state = setup("ffd")
    form_ffd = ra.Form(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    form_ffd.run()

    options, stochastic_model, limit_state = setup("ddm")
    form_ddm = ra.Form(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    form_ddm.run()

    assert pytest.approx(form_ffd.beta, abs=1e-5) == form_ddm.beta


def test_ddm_cmc():
    """
    Perform Crude Monte Carlo Simulation
    """
    options, stochastic_model, limit_state = setup("ddm")

    Analysis = ra.CrudeMonteCarlo(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert Analysis.x.shape[-1] == 1000


def test_ddm_is():
    """
    Perform Importance Sampling
    """
    options, stochastic_model, limit_state = setup("ddm")

    Analysis = ra.ImportanceSampling(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert Analysis.x.shape[-1] == 1000
