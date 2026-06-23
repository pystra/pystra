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
    assert np.isscalar(Analysis.beta)
    assert np.isscalar(Analysis.Pf)


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


def test_cmc_x_all_stores_physical_space():
    """Regression test for issue #90: x_all should store physical-space inputs.

    When multiple blocks are processed, x_all should contain values from the
    physical space (original distributions), not the Gaussian space.
    """
    options = ra.AnalysisOptions()
    options.setPrintOutput(False)
    options.setSamples(100)  # small for speed
    options.setBlockSize(30)  # force multiple blocks (100/30 = 4 blocks)

    # Use non-normal distributions with clear physical-space properties
    model = ra.model.StochasticModel()
    model.addVariable(ra.Lognormal("X1", 100, 20))  # positive values, mean ~100
    model.addVariable(ra.Uniform("X2", 10, 5))     # mean=10, bounds ~[1.34, 18.66]

    limit_state = ra.model.LimitState(lambda X1, X2: X1 - X2 - 100)

    Analysis = ra.CrudeMonteCarlo(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    Analysis.run()

    # x_all is stored as flat array (nrv * samples)
    nrv = 2
    samples = 100
    block_size = 30
    assert Analysis.x_all.shape == (nrv * samples,)

    # Extract X1 and X2 values from the flat array
    # Structure: [X1_block1, X2_block1, X1_block2, X2_block2, ...]
    x1_values = []
    x2_values = []
    for block in range(samples // block_size):
        base = block * nrv * block_size
        x1_values.extend(Analysis.x_all[base:base + block_size])
        x2_values.extend(Analysis.x_all[base + block_size:base + 2*block_size])
    x1_values = np.array(x1_values)
    x2_values = np.array(x2_values)

    # Physical-space checks:
    # Lognormal: all values should be positive (Gaussian space can be negative)
    assert np.all(x1_values > 0)

    # Uniform: values should be within bounds [1.34, 18.66]
    assert np.all(x2_values >= 1.0)
    assert np.all(x2_values <= 19.0)

    # Mean should be close to expected physical-space values
    # Lognormal(100, 20) has mean ≈ 100 * exp(0.5 * (20/100)^2) ≈ 100.2
    assert 90 < np.mean(x1_values) < 110

    # Uniform(10, 5) has mean = 10
    assert 8 < np.mean(x2_values) < 12


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
