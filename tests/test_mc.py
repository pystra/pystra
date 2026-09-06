"""Probability/index consistency at the boundaries of a simulation estimate."""

import numpy as np
import pytest

import pystra as ra


@pytest.mark.parametrize("value, probability, beta", [(1, 0, np.inf), (-1, 1, -np.inf)])
def test_monte_carlo_probability_boundary(value, probability, beta):
    model = ra.StochasticModel()
    model.addVariable(ra.Normal("X", 0, 1))
    options = ra.AnalysisOptions()
    options.setSamples(100)
    options.setBlockSize(25)
    analysis = ra.CrudeMonteCarlo(
        stochastic_model=model,
        limit_state=ra.LimitState(lambda X: value + 0 * X),
        analysis_options=options,
    )
    analysis.run()

    assert analysis.k == 100
    assert analysis.getFailure() == probability
    assert analysis.getBeta() == beta
