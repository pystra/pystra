import pytest
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
