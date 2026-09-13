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
    options = ra.FORMOptions()

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
        options=options,
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert pytest.approx(Analysis._beta, abs=1e-4) == 3.7347
    assert np.isscalar(Analysis._beta)
    assert np.isscalar(Analysis._Pf)


def test_form_svd():
    """
    Perform FORM analysis using SVD transform
    """
    options, stochastic_model, limit_state = setup()
    options = ra.FORMOptions(transform="svd")

    Analysis = ra.FORM(
        options=options,
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert pytest.approx(Analysis._beta, abs=1e-4) == 3.7347


def test_sorm():
    """
    Perform SORM analysis
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.SORM(
        options=ra.SORMOptions(form=options),
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert pytest.approx(Analysis._betaHL, abs=1e-4) == 3.7347
    # An independent u-space central-difference calculation gives 3.85389. The
    # earlier reference, 3.8537, came from code that transformed SORM gradients
    # at points shifted by the finite-difference step.
    assert pytest.approx(Analysis._betag_breitung, abs=1e-4) == 3.8539
    assert pytest.approx(Analysis._betag_breitung_m, abs=2e-4) == 3.8582


def test_sorm_pointfit():
    """
    SORM point-fitting analysis on the standard test problem.
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.SORM(
        model=stochastic_model,
        limit_state=limit_state,
        options=ra.SORMOptions(fit="point", form=options),
    )
    Analysis.run()

    # betaHL should match FORM
    assert pytest.approx(Analysis._betaHL, abs=1e-4) == 3.7347

    # Point-fitting gives similar but not identical results to curve-fitting
    assert pytest.approx(Analysis._betag_breitung, abs=5e-2) == 3.79
    assert pytest.approx(Analysis._betag_breitung_m, abs=5e-2) == 3.79

    # Asymmetric curvatures should be populated
    assert Analysis._kappa_pf is not None
    assert Analysis._kappa_pf.shape == (2, 2)  # 2 sides x (nrv-1) axes
    assert Analysis._fit_type == "pf"

    # Average curvatures stored in kappa for compatibility
    assert len(Analysis._kappa) == 2


def test_sorm_pointfit_linear():
    """
    Point-fitting SORM on a linear LSF should give betag == betaHL,
    since a linear surface has zero curvature everywhere.
    """
    options = ra.FORMOptions()

    model = ra.model.StochasticModel()
    model.add_variable(ra.Normal("R", 10, 2))
    model.add_variable(ra.Normal("S", 5, 1))

    limit_state = ra.model.LimitState(lambda R, S: R - S)

    Analysis = ra.SORM(
        model=model,
        limit_state=limit_state,
        options=ra.SORMOptions(fit="point", form=options),
    )
    Analysis.run()

    expected_beta = 5.0 / np.sqrt(5.0)
    assert pytest.approx(Analysis._betag_breitung, abs=1e-3) == expected_beta
    # Curvatures should be effectively zero
    assert np.allclose(Analysis._kappa_pf, 0, atol=1e-6)


def test_sorm_pointfit_with_form():
    """
    Pass a pre-computed FORM result to SORM point-fitting.
    """
    options, stochastic_model, limit_state = setup()

    form = ra.FORM(
        options=options,
        model=stochastic_model,
        limit_state=limit_state,
    )
    form.run()

    Analysis = ra.SORM(
        model=stochastic_model,
        limit_state=limit_state,
        form=form,
        options=ra.SORMOptions(fit="point", form=options),
    )
    Analysis.run()

    assert pytest.approx(Analysis._betaHL, abs=1e-4) == 3.7347
    assert Analysis._betag_breitung > 0
    assert Analysis._kappa_pf is not None


def test_sorm_invalid_fit_type():
    """
    An unknown fit is rejected when the options are created.
    """
    with pytest.raises(ValueError, match="fit must be one of"):
        ra.SORMOptions(fit="invalid")


def test_cmc():
    """
    Perform Crude Monte Carlo Simulation
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.CrudeMonteCarlo(
        options=ra.SimulationOptions(n_samples=1000),
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert Analysis._x.shape[-1] == 1000
    # beta should be non-negative
    assert Analysis._beta >= 0


def test_cmc_x_all_stores_physical_space():
    """Issue #90: every stored block must match the physical model inputs."""
    options = ra.SimulationOptions(n_samples=100, block_size=30, target_cov=0.0)
    # retain all four blocks, including the final ten

    model = ra.StochasticModel()
    model.add_variable(ra.Lognormal("X1", 100, 20))
    model.add_variable(ra.Uniform("X2", 10, 5))
    evaluated_blocks = []

    def limit_state(X1, X2):
        evaluated_blocks.append(np.array([X1, X2], copy=True))
        return X1 - X2 - 100

    analysis = ra.CrudeMonteCarlo(
        options=options,
        model=model,
        limit_state=ra.LimitState(limit_state),
        rng=90,
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
    np.testing.assert_array_equal(analysis._x_all, expected)
    assert analysis._x_all.shape == (200,)
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
        options=ra.SimulationOptions(n_samples=1000),
        model=stochastic_model,
        limit_state=limit_state,
    )
    # Initialise just enough internal state to call the method
    samples = 10
    Analysis._block_size = samples
    Analysis._q_bar = np.empty(samples)
    Analysis._cov_q_bar = np.empty(samples)

    # Force the zero-CoV branch: sum_q > 0 but all q values identical
    # so variance is exactly zero → cov_q_bar == 0
    Analysis._k = 5
    Analysis._sum_q = 5.0
    Analysis._log_sum_q = np.log(5.0)
    Analysis._log_sum_q2 = np.log(5.0)  # same as log sum_q → variance = 0
    Analysis._log_q_bar = np.empty(samples)

    Analysis._compute_coefficient_of_variation()
    # Should reach cov_q_bar = 1.0 without AttributeError
    assert Analysis._cov_q_bar[4] == 1.0


def test_is():
    """
    Perform Importance Sampling
    """
    options, stochastic_model, limit_state = setup()

    Analysis = ra.ImportanceSampling(
        options=ra.SimulationOptions(n_samples=1000),
        model=stochastic_model,
        limit_state=limit_state,
    )
    Analysis.run()

    # validate results
    assert Analysis._x.shape[-1] == 1000
    # beta should be positive for importance sampling
    assert Analysis._beta > 0


def test_distribution_analysis():
    """
    Perform distribution analysis
    """

    options, stochastic_model, limit_state = setup()

    # Perform Distribution analysis
    Analysis = ra.DistributionAnalysis(
        options=ra.SimulationOptions(n_samples=1000),
        model=stochastic_model,
        limit_state=limit_state,
        rng=42,
    )
    Analysis.run()

    # validate results statistically
    # Agree with an independent 1000-sample estimate (mean 1.0284, standard
    # deviation 0.1562) within about four combined standard errors.
    assert pytest.approx(Analysis._all_G.mean(), abs=0.025) == 1.0284
    assert pytest.approx(Analysis._all_G.std(), abs=0.02) == 0.1562


def test_form_uncorrelated_normals():
    """
    FORM for simple R - S problem with known analytical beta.
    beta = (mu_R - mu_S) / sqrt(sigma_R^2 + sigma_S^2)
    """
    options = ra.FORMOptions()

    model = ra.model.StochasticModel()
    model.add_variable(ra.Normal("R", 10, 2))
    model.add_variable(ra.Normal("S", 5, 1))

    limit_state = ra.model.LimitState(lambda R, S: R - S)

    Analysis = ra.FORM(
        options=options,
        model=model,
        limit_state=limit_state,
    )
    Analysis.run()

    # Analytical: beta = (10 - 5) / sqrt(4 + 1) = 5 / sqrt(5) ≈ 2.2361
    expected_beta = 5.0 / np.sqrt(5.0)
    assert pytest.approx(Analysis._beta, abs=1e-3) == expected_beta


def test_form_with_gumbel():
    """
    FORM with Gumbel distribution.
    """
    options = ra.FORMOptions()

    model = ra.model.StochasticModel()
    model.add_variable(ra.Normal("R", 20, 3))
    model.add_variable(ra.Gumbel("S", 10, 2))

    limit_state = ra.model.LimitState(lambda R, S: R - S)

    Analysis = ra.FORM(
        options=options,
        model=model,
        limit_state=limit_state,
    )
    Analysis.run()

    # beta should be positive and reasonable
    assert Analysis._beta > 0
    assert Analysis._beta < 10


def test_sorm_curvatures_match_independent_central_differences():
    """SORM's curvature fit agrees with a u-space Hessian that needs no Jacobian."""
    options, stochastic_model, limit_state = setup()
    sorm = ra.SORM(
        options=ra.SORMOptions(form=options),
        model=stochastic_model,
        limit_state=limit_state,
    )
    sorm.run()
    form = sorm.form
    u0 = np.ravel(form._u)
    marg = stochastic_model.get_marginal_distributions()
    names = stochastic_model.get_variables()
    constants = stochastic_model.constants

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
    assert np.sort(np.ravel(sorm._kappa)) == pytest.approx(np.sort(kappa), abs=3e-4)
