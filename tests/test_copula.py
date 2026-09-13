"""Distribution identities and published transformation benchmarks."""

import numpy as np
import pytest
from scipy.stats import norm, t, expon, multivariate_normal, multivariate_t
from scipy.integrate import quad
import pystra as ra

R = np.array([[1, 0.4], [0.4, 1]])


def normals(copula):
    return ra.JointDistribution([ra.Normal("X", 0, 1), ra.Normal("Y", 0, 1)], copula)


@pytest.mark.parametrize(
    "copula",
    [
        ra.GaussianCopula(R),
        ra.StudentTCopula(R, 4),
        ra.FrankCopula(10),
        ra.FrankCopula(-5),
        ra.FrankCopula(0),
    ],
)
@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_conditional_roundtrip(copula, order):
    p = np.array([[0.05, 0.7], [0.8, 0.3], [0.5, 0.5]])
    q = copula.rosenblatt(p, order)
    np.testing.assert_allclose(copula.inverse_rosenblatt(q, order), p, atol=1e-10)
    np.testing.assert_allclose(copula.rosenblatt(p[0], order), q[0])


def test_gaussian_joint_equals_multivariate_normal():
    joint = normals(ra.GaussianCopula(R))
    x = np.array([[0.2, -0.5], [1.1, 0.7]])
    np.testing.assert_allclose(
        joint.pdf(x), multivariate_normal.pdf(x, cov=R), rtol=1e-12
    )
    assert joint.cdf([0, 0]) == pytest.approx(
        0.25 + np.arcsin(0.4) / (2 * np.pi), abs=2e-6
    )
    assert joint.cdf([np.inf, 0.4]) == pytest.approx(norm.cdf(0.4))
    assert joint.cdf([-np.inf, np.inf]) == 0


def test_t_joint_equals_multivariate_t():
    joint = ra.JointDistribution(
        [ra.ScipyDist("X", t(4)), ra.ScipyDist("Y", t(4))],
        ra.StudentTCopula(R, 4),
    )
    points = np.array([[0.4, -1], [2, 1.2]])
    np.testing.assert_allclose(
        joint.pdf(points), multivariate_t.pdf(points, shape=R, df=4), rtol=1e-9
    )
    assert joint.cdf([0, 0], maxpts=100000, random_state=123) == pytest.approx(
        0.25 + np.arcsin(0.4) / (2 * np.pi), abs=1e-5
    )


def test_t_rosenblatt_whitens_independent_reference_samples():
    matrix = np.array([[1, 0.4, -0.2], [0.4, 1, 0.3], [-0.2, 0.3, 1]])
    w = multivariate_t.rvs(
        shape=matrix, df=5, size=20000, random_state=np.random.default_rng(3)
    )
    cop = ra.StudentTCopula(matrix, 5)
    u = norm.ppf(cop.rosenblatt(t.cdf(w, 5), order=[2, 0, 1]))
    np.testing.assert_allclose(u.mean(axis=0), 0, atol=0.025)
    np.testing.assert_allclose(np.cov(u.T), np.eye(3), atol=0.035)
    # Uncorrelated squares help distinguish independence from spherical t.
    np.testing.assert_allclose(np.corrcoef((u * u).T), np.eye(3), atol=0.035)


def test_identity_t_copula_is_not_independent():
    gaussian = ra.GaussianCopula(np.eye(2))
    student = ra.StudentTCopula(np.eye(2), 3)
    assert gaussian.pdf([0.95, 0.95]) == pytest.approx(1)
    assert student.pdf([0.95, 0.95]) > 1.5


@pytest.mark.parametrize(
    "copula", [ra.GaussianCopula(R), ra.StudentTCopula(R, 4), ra.FrankCopula(10)]
)
@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_transform_jacobian_and_density(copula, order):
    joint = ra.JointDistribution(
        [ra.Lognormal("X", 3, 1), ra.Normal("Y", 2, 0.5)], copula
    )
    transform = joint.make_transformation("rosenblatt", order=order)
    u = np.array([0.7, -0.4])
    x = transform.u_to_x(u)
    np.testing.assert_allclose(transform.x_to_u(x), u, atol=1e-9)
    J = transform.jacobian_u_wrt_x(u, x)
    numeric = np.empty((2, 2))
    h = 1e-5
    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        numeric[:, j] = (transform.x_to_u(x + step) - transform.x_to_u(x - step)) / (
            2 * h
        )
    np.testing.assert_allclose(J, numeric, atol=2e-6, rtol=2e-5)
    assert joint.pdf(x) == pytest.approx(
        np.prod(norm.pdf(u)) * abs(np.linalg.det(J)), rel=2e-7
    )


def test_gaussian_nataf_matches_rosenblatt():
    joint = normals(ra.GaussianCopula(R))
    a = joint.make_transformation("nataf")
    b = joint.make_transformation("rosenblatt")
    for x in ([0.3, 1.2], [-2, -1], [8, 0]):
        np.testing.assert_allclose(a.x_to_u(x), b.x_to_u(x), atol=1e-12)
        np.testing.assert_allclose(a.u_to_x(a.x_to_u(x)), x, atol=1e-12)


def test_generalized_t_nataf_and_form_exact_halfspace():
    cop = ra.StudentTCopula(R, 4)
    joint = ra.JointDistribution(
        [ra.ScipyDist("X", t(4)), ra.ScipyDist("Y", t(4))], cop
    )
    tr = joint.make_transformation("nataf")
    x = np.array([1.0, -2.0])
    u = tr.x_to_u(x)
    np.testing.assert_allclose(u, np.linalg.solve(np.linalg.cholesky(R), x), atol=1e-10)
    np.testing.assert_allclose(tr.u_to_x(u), x, atol=1e-10)
    assert tr.standard_space == "student_t"
    opts = ra.FORMOptions(transform="nataf")
    form = ra.FORM(
        model=ra.StochasticModel(joint),
        options=opts,
        limit_state=ra.LimitState(lambda X, Y: 3 - X - Y),
    )
    result = form.run()
    beta = 3 / np.sqrt(2 + 2 * 0.4)
    assert form._beta == pytest.approx(beta, rel=2e-6)
    assert form._Pf == pytest.approx(t.sf(beta, 4), rel=2e-6)
    assert form._get_equivalent_beta() == pytest.approx(-norm.ppf(form._Pf))
    assert result.standard_space == "student_t"
    assert result.design_index == pytest.approx(beta, rel=2e-6)
    assert result.beta == pytest.approx(-norm.ppf(t.sf(beta, 4)), rel=2e-6)
    assert result.failure_probability == pytest.approx(t.sf(beta, 4), rel=2e-6)
    with pytest.raises(ValueError, match="normal space"):
        ra.StrongMaximumTest(form)
    for analysis in (
        ra.CrudeMonteCarlo(
            form.model,
            form.limit_state,
            options=ra.SimulationOptions(transform="nataf"),
        ),
        ra.SORM(form.model, form.limit_state, options=ra.SORMOptions(form=opts)),
    ):
        with pytest.raises(ValueError, match="normal space"):
            analysis.run()
    analysis = ra.SystemFORM(
        form.model,
        ra.SeriesSystem([ra.Component("a", form.limit_state)]),
        options=opts,
    )
    with pytest.raises(RuntimeError, match="normal space"):
        analysis.run()


def exponential_model(copula):
    return ra.StochasticModel(
        ra.JointDistribution(
            [
                ra.ScipyDist("X1", expon(scale=1)),
                ra.ScipyDist("X2", expon(scale=1 / 3)),
            ],
            copula,
        )
    )


def test_lebrun_dutfoy_frank_order_benchmark():
    # 2009, section 6: exponential rates 1,3; theta=10; 8X1+2X2<1.
    cop = ra.FrankCopula(10)
    results = []
    for order in ([0, 1], [1, 0]):
        opts = ra.FORMOptions(transform="rosenblatt", rosenblatt_order=order)
        f = ra.FORM(
            model=exponential_model(cop),
            options=opts,
            limit_state=ra.LimitState(lambda X1, X2: 8 * X1 + 2 * X2 - 1),
        )
        f.run()
        results.append(f._Pf)
    np.testing.assert_allclose(results, [0.107, 0.122], atol=0.00015)
    exact = quad(
        lambda x: np.exp(-x)
        * cop.rosenblatt([-np.expm1(-x), -np.expm1(-3 * (1 - 8 * x) / 2)])[1],
        0,
        1 / 8,
    )[0]
    assert exact == pytest.approx(0.1038, abs=0.00005)


def test_gaussian_order_preserves_form_probability():
    results = []
    for mode, order in [
        ("nataf", None),
        ("rosenblatt", [0, 1]),
        ("rosenblatt", [1, 0]),
    ]:
        opts = ra.FORMOptions(transform=mode, rosenblatt_order=order)
        f = ra.FORM(
            model=exponential_model(ra.GaussianCopula([[1, 0.5], [0.5, 1]])),
            limit_state=ra.LimitState(lambda X1, X2: 8 * X1 + 2 * X2 - 1),
            options=opts,
        )
        f.run()
        results.append(f._Pf)
    np.testing.assert_allclose(results, results[0], atol=1e-10)


def test_explicit_copula_does_not_reinterpret_latent_as_physical_correlation():
    model = exponential_model(ra.GaussianCopula(R))
    np.testing.assert_array_equal(
        ra.dependence.correlation.compute_modified_correlation_matrix(model), R
    )
    with pytest.raises(ValueError, match="Pearson"):
        model.get_correlation()
    model.set_correlation(np.eye(2))
    assert model.get_copula() is None
    np.testing.assert_array_equal(model.get_correlation(), np.eye(2))


def test_kendall_parameterization_and_immutable_matrix():
    cop = ra.StudentTCopula.from_kendall_tau([[1, 0.5], [0.5, 1]], df=4)
    assert cop.correlation[0, 1] == pytest.approx(np.sqrt(0.5))
    np.testing.assert_allclose(cop.kendall_tau, [[1, 0.5], [0.5, 1]])
    R = cop.correlation
    R[0, 1] = 0
    assert cop.correlation[0, 1] != 0


@pytest.mark.parametrize(
    "matrix",
    [
        [[1, 1], [1, 1]],
        [[1, 2], [2, 1]],
        [[1, 0.2], [0.3, 1]],
        [[2, 0], [0, 1]],
        [[1, np.nan], [np.nan, 1]],
        [],
    ],
)
def test_invalid_copula_matrix(matrix):
    with pytest.raises(ValueError):
        ra.GaussianCopula(matrix)


def test_dimension_and_continuity_validation():
    with pytest.raises(ValueError, match="dimension"):
        ra.JointDistribution([ra.Normal("X", 0, 1)], ra.GaussianCopula(R))
    with pytest.raises(ValueError, match="continuous"):
        ra.JointDistribution(
            [ra.ZeroInflated("X", ra.Normal("Z", 1, 1), 0.2)], ra.IndependentCopula(1)
        )
    with pytest.raises(ValueError, match="elliptical"):
        normals(ra.FrankCopula(10)).make_transformation("nataf")
    with pytest.raises(ValueError, match="permutation"):
        normals(ra.GaussianCopula(R)).make_transformation("rosenblatt", order=[0, 0])
    m = exponential_model(ra.FrankCopula(10))
    with pytest.raises(ValueError, match="before setting"):
        m.add_variable(ra.Normal("Z", 0, 1))
    m.add_variable(ra.Constant("C", 3))


def test_sampling_seed_and_joint_marginals():
    joint = normals(ra.StudentTCopula(R, 4))
    a = joint.rvs(10000, seed=3)
    b = joint.rvs(10000, seed=3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.mean(a, axis=0), 0, atol=0.025)
    np.testing.assert_allclose(np.std(a, axis=0), 1, atol=0.025)
    assert joint.rvs(0).shape == (0, 2)


@pytest.mark.parametrize("theta", [1e-200, 1e-12, 30, -30])
def test_frank_upper_corner_and_boundary(theta):
    cop = ra.FrankCopula(theta)
    np.testing.assert_allclose(
        cop.cdf([[0, 0.5], [1, 0.5], [1, 1]]), [0, 0.5, 1], atol=1e-13
    )
    p = np.array([[0.95, 0.99], [0.8, 0.85], [0.05, 0.1]])
    recovered = cop.inverse_rosenblatt(cop.rosenblatt(p))
    # At theta=-30 these upper-tail events have vanishing conditional density.
    # Uniform-score rounding is amplified by the inverse conditional derivative.
    tolerance = 1e-10 + 5 * np.finfo(float).eps / cop.pdf(p)
    assert np.all(np.abs(recovered - p) <= tolerance[:, None])


def test_frank_monte_carlo_integrates_original_event():
    m = exponential_model(ra.FrankCopula(10))
    opts = ra.SimulationOptions(n_samples=5000, target_cov=0)
    state = np.random.get_state()
    try:
        mc = ra.CrudeMonteCarlo(
            model=m,
            limit_state=ra.LimitState(lambda X1, X2: 8 * X1 + 2 * X2 - 1),
            options=opts,
            rng=20,
        )
        mc.run()
    finally:
        np.random.set_state(state)
    assert mc.transform.standard_space == "normal"
    assert abs(mc._Pf - 0.1038) < 5 * np.sqrt(0.1038 * (1 - 0.1038) / 5000)


def test_t_rosenblatt_supports_system_form_and_strong_maximum():
    model = ra.StochasticModel(normals(ra.StudentTCopula(R, 4)))
    system = ra.SeriesSystem(
        [ra.Component("a", lambda X: 3 - X), ra.Component("b", lambda X: 4 - X)]
    )
    analysis = ra.SystemFORM(model, system)
    analysis.run()
    assert analysis._Pf == pytest.approx(norm.sf(3), rel=1e-5)
    form = analysis._component_results["a"]
    assert form.transform.method == "rosenblatt"
    check = ra.StrongMaximumTest(form, point_number=100, rng=3)
    check.run()
    assert check._status == "no_competing_region_detected"


def test_numerical_sensitivity_preserves_spherical_t_options():
    model = ra.StochasticModel(
        ra.JointDistribution([ra.Normal("X", 0, 1)], ra.StudentTCopula([[1]], 4))
    )
    options = ra.FORMOptions(
        transform="nataf", limit_state_tolerance=1e-9, gradient_tolerance=1e-9
    )
    result = ra.SensitivityAnalysis(
        model=model,
        limit_state=ra.LimitState(lambda X: 3 - X),
        options=options,
        delta=1e-5,
    ).run()
    beta = t.isf(norm.sf(3), 4)
    derivative = -norm.pdf(3) / t.pdf(beta, 4)
    assert result.marginal["X"]["mean"] == pytest.approx(derivative, rel=1e-4)
    with pytest.raises(ValueError, match="legacy physical Pearson"):
        ra.SensitivityAnalysis(
            model=model,
            limit_state=ra.LimitState(lambda X: 3 - X),
            options=options,
            method="closed_form",
        ).run()


@pytest.mark.parametrize(
    "copula", [ra.GaussianCopula(R), ra.StudentTCopula(R, 4), ra.FrankCopula(10)]
)
@pytest.mark.parametrize("method, order", [("rosenblatt", [1, 0]), ("nataf", None)])
def test_directed_joint_jacobians(copula, method, order):
    if method == "nataf" and not copula.elliptical:
        return
    joint = ra.JointDistribution(
        [ra.Lognormal("X", 3, 1), ra.Normal("Y", 2, 0.5)], copula
    )
    transform = joint.make_transformation(method, order=order)
    u = np.array([0.7, -0.4])
    x = transform.u_to_x(u)
    inverse = transform.jacobian_x_wrt_u(u, x)
    steps = np.eye(2) * 1e-5
    numeric = np.column_stack(
        [
            (transform.u_to_x(u + step) - transform.u_to_x(u - step)) / 2e-5
            for step in steps
        ]
    )
    np.testing.assert_allclose(inverse, numeric, atol=2e-8)
    np.testing.assert_allclose(
        transform.jacobian_u_wrt_x(u, x) @ inverse, np.eye(2), atol=1e-12
    )
