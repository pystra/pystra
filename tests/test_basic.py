import pytest
import numpy as np
import pystra as ra


def lsf(X1, X2, X3):
    """
    example limit state function
    """
    return 1.7 - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2


def setup():
    """
    Set up simulation
    """
    # Set some options (optional)
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)
    options.setSamples(1000)  # only relevant for Monte Carlo

    # Set stochastic model
    stochastic_model = ra.model.StochasticModel()

    # Define random variables
    stochastic_model.addVariable(ra.Lognormal("X1", 500, 100))
    stochastic_model.addVariable(ra.Normal("X2", 2000, 400))
    stochastic_model.addVariable(ra.Uniform("X3", 5, 0.5))

    stochastic_model.setCorrelation(
        ra.correlation.CorrelationMatrix(
            [[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]]
        )
    )

    # Set limit state
    limit_state = ra.model.LimitState(lsf)

    return options, stochastic_model, limit_state


def test_form():
    """
    Perform FORM analysis
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.Form(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert pytest.approx(Analysis.beta, abs=1e-4) == 3.7347


def test_form_svd():
    """
    Perform FORM analysis using SVD transform
    """
    options, stochastic_model, limit_state = setup()
    options.setTransform("svd")

    Analysis = ra.Form(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert pytest.approx(Analysis.beta, abs=1e-4) == 3.7347


def test_sorm():
    """
    Perform SORM analysis
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.Sorm(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    print(Analysis.betag_breitung)
    print(Analysis.betag_breitung_m)

    # validate results
    assert pytest.approx(Analysis.betaHL, abs=1e-4) == 3.7347
    assert pytest.approx(Analysis.betag_breitung, abs=1e-4) == 3.8537
    assert pytest.approx(Analysis.betag_breitung_m, abs=2e-4) == 3.8582


def test_cmc():
    """
    Perform Crude Monte Carlo Simulation
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.CrudeMonteCarlo(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert Analysis.x.shape[-1] == 1000
    # beta should be non-negative
    assert Analysis.beta >= 0


def test_is():
    """
    Perform Importance Sampling
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.ImportanceSampling(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert Analysis.x.shape[-1] == 1000
    # beta should be positive for importance sampling
    assert Analysis.beta > 0


def test_distribution_analysis():
    """
    Perform distribution analysis
    """

    options, stochastic_model, limit_state = setup()
    options.print_output = False

    # Perform Distribution analysis
    Analysis = ra.DistributionAnalysis(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert pytest.approx(Analysis.all_G.mean(), abs=2e-2) == 1.03296
    assert pytest.approx(Analysis.all_G.std(), abs=1.5e-2) == 0.15989


def test_form_uncorrelated_normals():
    """
    FORM for simple R - S problem with known analytical beta.
    beta = (mu_R - mu_S) / sqrt(sigma_R^2 + sigma_S^2)
    """
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)

    model = ra.model.StochasticModel()
    model.addVariable(ra.Normal("R", 10, 2))
    model.addVariable(ra.Normal("S", 5, 1))

    limit_state = ra.model.LimitState(lambda R, S: R - S)

    Analysis = ra.Form(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    Analysis.run()

    # Analytical: beta = (10 - 5) / sqrt(4 + 1) = 5 / sqrt(5) ≈ 2.2361
    expected_beta = 5.0 / np.sqrt(5.0)
    assert pytest.approx(Analysis.beta, abs=1e-3) == expected_beta


def test_form_with_gumbel():
    """
    FORM with Gumbel distribution.
    """
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)

    model = ra.model.StochasticModel()
    model.addVariable(ra.Normal("R", 20, 3))
    model.addVariable(ra.Gumbel("S", 10, 2))

    limit_state = ra.model.LimitState(lambda R, S: R - S)

    Analysis = ra.Form(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    Analysis.run()

    # beta should be positive and reasonable
    assert Analysis.beta > 0
    assert Analysis.beta < 10
