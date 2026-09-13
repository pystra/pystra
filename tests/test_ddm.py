import pytest
import numpy as np

import pystra as ra


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

    options = ra.FORMOptions(differentiation=diff_mode)
    stochastic_model = ra.StochasticModel()

    # Define random variables
    stochastic_model.add_variable(ra.Lognormal("X1", 500, 100))
    stochastic_model.add_variable(ra.Lognormal("X2", 2000, 400))
    stochastic_model.add_variable(ra.Uniform("X3", 5, 0.5))
    stochastic_model.add_variable(ra.Lognormal("X4", 450, 90))
    stochastic_model.add_variable(ra.Lognormal("X5", 1800, 360))
    stochastic_model.add_variable(ra.Uniform("X6", 4.5, 0.45))

    # Define constants
    stochastic_model.add_variable(ra.Constant("r", 1.7))

    stochastic_model.set_correlation(
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
    form_ffd = ra.FORM(
        options=options,
        model=stochastic_model,
        limit_state=limit_state,
    )
    form_ffd.run()

    options, stochastic_model, limit_state = setup("ddm")
    form_ddm = ra.FORM(
        options=options,
        model=stochastic_model,
        limit_state=limit_state,
    )
    form_ddm.run()

    assert pytest.approx(form_ffd._beta, abs=1e-5) == form_ddm._beta


def test_ddm_cmc():
    """
    Perform Crude Monte Carlo Simulation
    """
    options, stochastic_model, limit_state = setup("ddm")

    Analysis = ra.CrudeMonteCarlo(
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert Analysis._x.shape[-1] == 1000


def test_ddm_is():
    """
    Perform Importance Sampling
    """
    options, stochastic_model, limit_state = setup("ddm")

    # Importance sampling samples about a FORM point; run FORM with DDM.
    form = ra.FORM(stochastic_model, limit_state, options=options)
    form.run()
    Analysis = ra.ImportanceSampling(
        model=stochastic_model,
        limit_state=limit_state,
        form=form,
    )
    Analysis.run()

    # validate results
    assert Analysis._x.shape[-1] == 1000
