import pytest
import numpy as np
import pystra as ra


@pytest.fixture
def simple_model():
    """3-variable stochastic model with correlation."""
    model = ra.model.StochasticModel()
    model.add_variable(ra.Lognormal("X1", 500, 100))
    model.add_variable(ra.Normal("X2", 2000, 400))
    model.add_variable(ra.Uniform("X3", 5, 0.5))
    model.set_correlation(
        ra.dependence.correlation.CorrelationMatrix(
            [[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]]
        )
    )
    return model


@pytest.fixture
def uncorrelated_model():
    """2-variable uncorrelated stochastic model."""
    model = ra.model.StochasticModel()
    model.add_variable(ra.Normal("R", 10, 2))
    model.add_variable(ra.Normal("S", 5, 1))
    return model


@pytest.fixture
def simple_limit_state():
    """Basic limit-state function: g = X1 - X2."""

    def lsf(R, S):
        return R - S

    return ra.model.LimitState(lsf)


@pytest.fixture
def analysis_options():
    """Default FORM settings."""
    opts = ra.FORMOptions()
    return opts
