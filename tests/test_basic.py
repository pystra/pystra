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
    options.set_print_output(False)
    options.set_samples(1000)  # only relevant for Monte Carlo

    # Set stochastic model
    stochastic_model = ra.model.StochasticModel()

    # Define random variables
    stochastic_model.add_variable(ra.Lognormal("X1", 500, 100))
    stochastic_model.add_variable(ra.Normal("X2", 2000, 400))
    stochastic_model.add_variable(ra.Uniform("X3", 5, 0.5))

    stochastic_model.set_correlation(
        ra.dependence.correlation.CorrelationMatrix(
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

    Analysis = ra.FORM(
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
    options.set_transform("svd")

    Analysis = ra.FORM(
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

    Analysis = ra.SORM(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    print(Analysis.betag_breitung)
    print(Analysis.betag_breitung_m)

    # validate results
    assert pytest.approx(Analysis.betaHL, abs=1e-4) == 3.7347
    # An independent u-space central-difference calculation gives 3.85389. The
    # earlier reference, 3.8537, came from code that transformed SORM gradients
    # at points shifted by the finite-difference step.
    assert pytest.approx(Analysis.betag_breitung, abs=1e-4) == 3.8539
    assert pytest.approx(Analysis.betag_breitung_m, abs=2e-4) == 3.8582


def test_sorm_pointfit():
    """
    SORM point-fitting analysis on the standard test problem.
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.SORM(
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
    options.set_print_output(False)

    model = ra.model.StochasticModel()
    model.add_variable(ra.Normal("R", 10, 2))
    model.add_variable(ra.Normal("S", 5, 1))

    limit_state = ra.model.LimitState(lambda R, S: R - S)

    Analysis = ra.SORM(
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
    Pass a pre-computed FORM result to SORM point-fitting.
    """
    options, stochastic_model, limit_state = setup()

    form = ra.FORM(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    form.run()

    Analysis = ra.SORM(
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

    Analysis = ra.SORM(
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


def test_cmc_x_all_stores_physical_space(monkeypatch):
    """Issue #90: every stored block must match the physical model inputs."""
    options = ra.AnalysisOptions()
    options.set_print_output(False)
    options.set_samples(100)
    options.set_block_size(30)
    options.target_cov = 0.0  # retain all four blocks, including the final ten

    # The legacy Monte Carlo runner uses NumPy's global random interface.
    rng = np.random.default_rng(90)
    monkeypatch.setattr(np.random, "randn", lambda *shape: rng.standard_normal(shape))

    model = ra.StochasticModel()
    model.add_variable(ra.Lognormal("X1", 100, 20))
    model.add_variable(ra.Uniform("X2", 10, 5))
    evaluated_blocks = []

    def limit_state(X1, X2):
        evaluated_blocks.append(np.array([X1, X2], copy=True))
        return X1 - X2 - 100

    analysis = ra.CrudeMonteCarlo(
        analysis_options=options,
        stochastic_model=model,
        limit_state=ra.LimitState(limit_state),
    )
    analysis.run()

    assert [block.shape for block in evaluated_blocks] == [
        (2, 30),
        (2, 30),
        (2, 30),
        (2, 10),
    ]
    # x_all is block-major: X1 then X2 within each block. Compare the actual
    # evaluated inputs, rather than inferring coordinates from sample means.
    expected = np.concatenate([block.ravel() for block in evaluated_blocks])
    np.testing.assert_array_equal(analysis.x_all, expected)
    assert analysis.x_all.shape == (200,)
    physical = np.concatenate(evaluated_blocks, axis=1)
    assert np.all(physical[0] > 0)
    assert np.all(physical[1] >= 10 - np.sqrt(3) * 5)
    assert np.all(physical[1] <= 10 + np.sqrt(3) * 5)


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

    Analysis.compute_coefficient_of_variation()
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
    options.set_print_output(False)

    model = ra.model.StochasticModel()
    model.add_variable(ra.Normal("R", 10, 2))
    model.add_variable(ra.Normal("S", 5, 1))

    limit_state = ra.model.LimitState(lambda R, S: R - S)

    Analysis = ra.FORM(
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
    options.set_print_output(False)

    model = ra.model.StochasticModel()
    model.add_variable(ra.Normal("R", 20, 3))
    model.add_variable(ra.Gumbel("S", 10, 2))

    limit_state = ra.model.LimitState(lambda R, S: R - S)

    Analysis = ra.FORM(
        analysis_options=options,
        stochastic_model=model,
        limit_state=limit_state,
    )
    Analysis.run()

    # beta should be positive and reasonable
    assert Analysis.beta > 0
    assert Analysis.beta < 10


def test_sorm_curvatures_match_independent_central_differences():
    """SORM's curvature fit agrees with a u-space Hessian that needs no Jacobian."""
    options, stochastic_model, limit_state = setup()
    sorm = ra.SORM(
        analysis_options=options,
        stochastic_model=stochastic_model,
        limit_state=limit_state,
    )
    sorm.run()
    form = sorm.form
    u0 = np.ravel(form.get_design_point())
    marg = stochastic_model.get_marginal_distributions()
    names = stochastic_model.get_variables()
    constants = stochastic_model.get_constants()

    def g(u):
        x = np.ravel(form.transform.u_to_x(u, marg))
        return float(
            np.ravel(limit_state.expression(**dict(zip(names, x)), **constants))[0]
        )

    n, d, eye = len(u0), 1e-3, np.eye(len(u0))
    grad = np.array([(g(u0 + d * e) - g(u0 - d * e)) / (2 * d) for e in eye])
    hess = np.array(
        [
            [
                (
                    g(u0 + d * (eye[i] + eye[j]))
                    - g(u0 + d * (eye[i] - eye[j]))
                    - g(u0 - d * (eye[i] - eye[j]))
                    + g(u0 - d * (eye[i] + eye[j]))
                )
                / (4 * d * d)
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    basis = np.linalg.qr(np.column_stack([u0 / np.linalg.norm(u0), eye]))[0][:, 1:n]
    kappa = np.linalg.eigvalsh(basis.T @ hess @ basis) / np.linalg.norm(grad)
    # The earlier gradient transformation at shifted points was off by 7.5e-4.
    assert np.sort(np.ravel(sorm.kappa)) == pytest.approx(np.sort(kappa), abs=3e-4)
