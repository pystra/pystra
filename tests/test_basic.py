import pytest
import pystra as ra


def example_limitstatefunction(X1, X2, X3):
    """
    example limit state function
    """
    return 1 - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2


def setup():
    """
    Set up simulation
    """
    g = example_limitstatefunction

    # Set some options (optional)
    options = ra.model.AnalysisOptions()
    options.printResults(False)
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
    limit_state = ra.model.LimitState(g)

    return g, options, stochastic_model, limit_state


def test_form():
    """
    Perform FORM analysis
    """
    g, options, stochastic_model, limit_state = setup()

    Analysis = ra.Form(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )

    # validate results
    assert Analysis.i == 17
    assert pytest.approx(Analysis.beta, abs=1e-3) == 1.753


def test_cmc():
    """
    Perform Crude Monte Carlo Simulation
    """
    g, options, stochastic_model, limit_state = setup()

    Analysis = ra.CrudeMonteCarlo(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )

    # validate results
    assert Analysis.x.shape[-1] == 1000


def test_is():
    """
    Perform Importance Sampling
    """
    g, options, stochastic_model, limit_state = setup()

    Analysis = ra.ImportanceSampling(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )

    # validate results
    assert Analysis.x.shape[-1] == 1000
