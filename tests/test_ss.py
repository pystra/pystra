"""Tests for Subset Simulation (SubsetSimulation)."""

import numpy as np
import pytest
import pystra as ra


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def simple_rs_model():
    """R - S problem with N(10, 2) and N(5, 1).

    Analytical beta = (10 - 5) / sqrt(4 + 1) = 5/sqrt(5) ≈ 2.2361.
    """
    model = ra.StochasticModel()
    model.addVariable(ra.Normal("R", 10, 2))
    model.addVariable(ra.Normal("S", 5, 1))
    limit_state = ra.LimitState(lambda R, S: R - S)
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)
    return model, limit_state, options


def standard_3rv_model():
    """Three-variable correlated problem used throughout test_basic.py.

    FORM beta ≈ 3.7347.
    """

    def lsf(X1, X2, X3):
        return 1.7 - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2

    model = ra.StochasticModel()
    model.addVariable(ra.Lognormal("X1", 500, 100))
    model.addVariable(ra.Normal("X2", 2000, 400))
    model.addVariable(ra.Uniform("X3", 5, 0.5))
    model.setCorrelation(
        ra.CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
    )
    limit_state = ra.LimitState(lsf)
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)
    return model, limit_state, options


# ---------------------------------------------------------------------------
# Subset Simulation tests
# ---------------------------------------------------------------------------


def test_ss_simple_rs():
    """Subset Simulation on R - S problem should give beta close to analytical."""
    np.random.seed(42)
    model, limit_state, options = simple_rs_model()
    options.setSamples(1000)

    analysis = ra.SubsetSimulation(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
        p0=0.1,
    )
    analysis.run()

    analytical_beta = 5.0 / np.sqrt(5.0)  # ≈ 2.2361
    assert analysis.beta > 0
    assert pytest.approx(analysis.beta, abs=0.5) == analytical_beta
    assert 0.0 < analysis.Pf < 1.0
    assert analysis.cov >= 0.0


def test_ss_thresholds_decreasing():
    """Intermediate thresholds must be non-increasing, ending at 0."""
    np.random.seed(5)
    model, limit_state, options = simple_rs_model()
    options.setSamples(500)

    analysis = ra.SubsetSimulation(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    thresholds = analysis.thresholds
    assert len(thresholds) >= 1
    assert thresholds[-1] == 0.0
    for i in range(1, len(thresholds)):
        assert thresholds[i] <= thresholds[i - 1]


def test_ss_n_levels_consistent():
    """n_levels must equal the number of thresholds and conditional_probs."""
    np.random.seed(3)
    model, limit_state, options = simple_rs_model()
    options.setSamples(500)

    analysis = ra.SubsetSimulation(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    assert analysis.n_levels == len(analysis.thresholds)
    assert analysis.n_levels == len(analysis.conditional_probs)


def test_ss_conditional_probs_valid():
    """Conditional probabilities at each level must lie in (0, 1]."""
    np.random.seed(11)
    model, limit_state, options = simple_rs_model()
    options.setSamples(500)

    analysis = ra.SubsetSimulation(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    for p in analysis.conditional_probs:
        assert 0.0 < p <= 1.0


def test_ss_standard_problem():
    """Subset Simulation on the three-variable correlated problem."""
    np.random.seed(77)
    model, limit_state, options = standard_3rv_model()
    options.setSamples(1000)

    analysis = ra.SubsetSimulation(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
        p0=0.1,
    )
    analysis.run()

    # FORM beta ≈ 3.7347; SS should be in a reasonable neighbourhood
    assert analysis.beta > 0
    assert pytest.approx(analysis.beta, abs=0.7) == 3.7347


def test_ss_invalid_p0():
    """p0 outside (0, 1) must raise ValueError."""
    model, limit_state, options = simple_rs_model()
    with pytest.raises(ValueError, match="p0 must be in"):
        ra.SubsetSimulation(
            analysis_options=options,
            stochastic_model=model,
            limit_state=limit_state,
            p0=1.5,
        )


def test_ss_product_of_cond_probs():
    """Pf must equal the product of the conditional probabilities."""
    np.random.seed(21)
    model, limit_state, options = simple_rs_model()
    options.setSamples(500)

    analysis = ra.SubsetSimulation(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    expected = float(np.prod(analysis.conditional_probs))
    assert pytest.approx(analysis.Pf, rel=1e-12) == expected


def test_ss_results_valid_flag():
    """results_valid should be True after run()."""
    np.random.seed(0)
    model, limit_state, options = simple_rs_model()
    options.setSamples(200)

    analysis = ra.SubsetSimulation(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    assert not analysis.results_valid
    analysis.run()
    assert analysis.results_valid


def test_ss_get_beta_get_failure_consistent():
    """getBeta() and getFailure() should be consistent with stored attributes."""
    np.random.seed(8)
    model, limit_state, options = simple_rs_model()
    options.setSamples(300)

    analysis = ra.SubsetSimulation(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    assert analysis.getBeta() == analysis.beta
    assert analysis.getFailure() == analysis.Pf


def test_ss_custom_proposal_sigma():
    """A non-default proposal_sigma should still give valid results."""
    np.random.seed(17)
    model, limit_state, options = simple_rs_model()
    options.setSamples(500)

    analysis = ra.SubsetSimulation(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
        proposal_sigma=0.5,
    )
    analysis.run()

    assert analysis.beta > 0
    assert 0.0 < analysis.Pf < 1.0
