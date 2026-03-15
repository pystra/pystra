"""Tests for Line Sampling (LineSampling)."""

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
# Line Sampling tests
# ---------------------------------------------------------------------------


def test_ls_simple_rs():
    """Line Sampling on R - S problem should give beta close to analytical."""
    np.random.seed(42)
    model, limit_state, options = simple_rs_model()
    options.setSamples(2000)

    analysis = ra.LineSampling(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    analytical_beta = 5.0 / np.sqrt(5.0)  # ≈ 2.2361
    assert analysis.beta > 0
    assert pytest.approx(analysis.beta, abs=0.3) == analytical_beta
    assert 0.0 < analysis.Pf < 1.0
    assert analysis.cov >= 0.0
    assert analysis.n_samples == 2000


def test_ls_alpha_is_unit_vector():
    """After run(), alpha must be a unit vector."""
    np.random.seed(0)
    model, limit_state, options = simple_rs_model()
    options.setSamples(500)

    analysis = ra.LineSampling(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    assert pytest.approx(np.linalg.norm(analysis.alpha), abs=1e-10) == 1.0


def test_ls_with_precomputed_form():
    """Line Sampling accepts a pre-computed FORM result."""
    np.random.seed(7)
    model, limit_state, options = simple_rs_model()
    options.setSamples(1000)

    form = ra.Form(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    form.run()

    analysis = ra.LineSampling(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
        form=form,
    )
    analysis.run()

    # Should share the same alpha as FORM
    assert np.allclose(analysis.alpha, form.getAlpha())
    assert analysis.beta > 0


def test_ls_standard_problem():
    """Line Sampling on the three-variable correlated problem."""
    np.random.seed(13)
    model, limit_state, options = standard_3rv_model()
    options.setSamples(1000)

    analysis = ra.LineSampling(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    # FORM beta ≈ 3.7347; LS should be in a reasonable neighbourhood
    assert analysis.beta > 0
    assert pytest.approx(analysis.beta, abs=0.5) == 3.7347


def test_ls_pf_contributions_shape():
    """Internal pf_contributions array should have length N with values in [0,1]."""
    np.random.seed(99)
    model, limit_state, options = simple_rs_model()
    N = 200
    options.setSamples(N)

    analysis = ra.LineSampling(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    assert analysis._pf_contributions is not None
    assert len(analysis._pf_contributions) == N
    assert np.all(analysis._pf_contributions >= 0.0)
    assert np.all(analysis._pf_contributions <= 1.0)


def test_ls_results_valid_flag():
    """results_valid should be True after run()."""
    np.random.seed(1)
    model, limit_state, options = simple_rs_model()
    options.setSamples(100)

    analysis = ra.LineSampling(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    assert not analysis.results_valid
    analysis.run()
    assert analysis.results_valid


def test_ls_get_beta_get_failure_consistent():
    """getBeta() and getFailure() should be consistent with stored attributes."""
    np.random.seed(5)
    model, limit_state, options = simple_rs_model()
    options.setSamples(500)

    analysis = ra.LineSampling(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    analysis.run()

    assert analysis.getBeta() == analysis.beta
    assert analysis.getFailure() == analysis.Pf
