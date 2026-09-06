"""Probability/index consistency at the boundaries of a simulation estimate."""

import numpy as np
import pytest

import pystra as ra


@pytest.mark.parametrize("value, probability, beta", [(1, 0, np.inf), (-1, 1, -np.inf)])
def test_monte_carlo_probability_boundary(value, probability, beta):
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    options = ra.AnalysisOptions()
    options.set_samples(100)
    options.set_block_size(25)
    analysis = ra.CrudeMonteCarlo(
        stochastic_model=model,
        limit_state=ra.LimitState(lambda X: value + 0 * X),
        analysis_options=options,
    )
    analysis.run()

    assert analysis.k == 100
    assert analysis.get_failure() == probability
    assert analysis.get_beta() == beta
