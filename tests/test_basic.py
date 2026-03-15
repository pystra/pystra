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


def test_sorm_pointfit():
    """
    SORM point-fitting analysis on the standard test problem.
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.Sorm(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run(fit_type="pf")

    # betaHL should match FORM
    assert pytest.approx(Analysis.betaHL, abs=1e-4) == 3.7347

    # Point-fitting gives similar but not identical results to curve-fitting
    assert pytest.approx(Analysis.betag_breitung, abs=5e-2) == 3.79
    assert pytest.approx(Analysis.betag_breitung_m, abs=5e-2) == 3.79

    # Asymmetric curvatures should be populated
    assert Analysis.kappa_pf is not None
    assert Analysis.kappa_pf.shape == (2, 2)  # 2 sides x (nrv-1) axes
    assert Analysis.fit_type == "pf"

    # Average curvatures stored in kappa for compatibility
    assert len(Analysis.kappa) == 2


def test_sorm_pointfit_linear():
    """
    Point-fitting SORM on a linear LSF should give betag == betaHL,
    since a linear surface has zero curvature everywhere.
    """
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)

    model = ra.model.StochasticModel()
    model.addVariable(ra.Normal("R", 10, 2))
    model.addVariable(ra.Normal("S", 5, 1))

    limit_state = ra.model.LimitState(lambda R, S: R - S)

    Analysis = ra.Sorm(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    Analysis.run(fit_type="pf")

    expected_beta = 5.0 / np.sqrt(5.0)
    assert pytest.approx(Analysis.betag_breitung, abs=1e-3) == expected_beta
    # Curvatures should be effectively zero
    assert np.allclose(Analysis.kappa_pf, 0, atol=1e-6)


def test_sorm_pointfit_with_form():
    """
    Pass a pre-computed Form result to SORM point-fitting.
    """
    options, stochastic_model, limit_state = setup()

    form = ra.Form(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    form.run()

    Analysis = ra.Sorm(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
        form=form,
    )
    Analysis.run(fit_type="pf")

    assert pytest.approx(Analysis.betaHL, abs=1e-4) == 3.7347
    assert Analysis.betag_breitung > 0
    assert Analysis.kappa_pf is not None


def test_sorm_invalid_fit_type():
    """
    Invalid fit_type should raise ValueError.
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.Sorm(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    with pytest.raises(ValueError, match="Unknown fit_type"):
        Analysis.run(fit_type="invalid")


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


def test_mc_cov_zero_branch():
    """Regression test for issue #64: cov_of_q_bar typo.

    When the computed CoV is exactly zero the MC code should set
    cov_q_bar = 1.0 without raising AttributeError.
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.CrudeMonteCarlo(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    # Initialise just enough internal state to call the method
    samples = 10
    Analysis.block_size = samples
    Analysis.q_bar = np.empty(samples)
    Analysis.cov_q_bar = np.empty(samples)

    # Force the zero-CoV branch: sum_q > 0 but all q values identical
    # so variance is exactly zero → cov_q_bar == 0
    Analysis.k = 5
    Analysis.sum_q = 5.0
    Analysis.sum_q2 = 5.0  # same as sum_q → variance = 0

    Analysis.computeCoefficientOfVariation()
    # Should reach cov_q_bar = 1.0 without AttributeError
    assert Analysis.cov_q_bar[4] == 1.0


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

    np.random.seed(42)

    # Perform Distribution analysis
    Analysis = ra.DistributionAnalysis(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results (fixed seed=42 gives deterministic output)
    assert pytest.approx(Analysis.all_G.mean(), abs=1e-6) == 1.02840644
    assert pytest.approx(Analysis.all_G.std(), abs=1e-6) == 0.15620518


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
