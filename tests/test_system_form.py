"""Analytical probability benchmarks for component-based system FORM."""

import numpy as np
import pytest
from scipy.stats import norm
import pystra as ra


def model2(rho=0):
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    model.add_variable(ra.Normal("Y", 0, 1))
    model.set_correlation([[1, rho], [rho, 1]])
    return model


@pytest.mark.parametrize("kind", [ra.SeriesSystem, ra.ParallelSystem])
@pytest.mark.parametrize("beta", [2.0, 3.0, 8.0])
def test_independent_exact(kind, beta):
    system = kind(
        [ra.Component("x", lambda X: beta - X), ra.Component("y", lambda Y: beta - Y)]
    )
    analysis = ra.SystemForm(system, model2())
    analysis.run()
    p = norm.sf(beta)
    exact = 2 * p - p * p if kind is ra.SeriesSystem else p * p
    assert analysis.get_failure() == pytest.approx(exact, rel=1e-6, abs=0)
    assert analysis.get_beta() == pytest.approx(-norm.ppf(exact), rel=1e-6)
    np.testing.assert_allclose(analysis.correlation, np.eye(2), atol=1e-12)
    assert all(f.converged for f in analysis.component_results.values())


@pytest.mark.parametrize("kind", [ra.SeriesSystem, ra.ParallelSystem])
@pytest.mark.parametrize("rho", [-0.7, 0.6])
def test_correlated_orthant(kind, rho):
    # P(X>0,Y>0) = 1/4 + asin(rho)/(2*pi).
    system = kind([ra.Component("x", lambda X: -X), ra.Component("y", lambda Y: -Y)])
    analysis = ra.SystemForm(system, model2(rho))
    analysis.run()
    joint = 0.25 + np.arcsin(rho) / (2 * np.pi)
    exact = 1 - joint if kind is ra.SeriesSystem else joint
    assert analysis.get_failure() == pytest.approx(exact, abs=2e-7)
    assert analysis.correlation[0, 1] == pytest.approx(rho)


@pytest.mark.parametrize("kind", [ra.SeriesSystem, ra.ParallelSystem])
@pytest.mark.parametrize("opposing", [False, True])
def test_singular_component_directions(kind, opposing):
    system = kind(
        [
            ra.Component("a", lambda X: 3 - X),
            ra.Component("b", lambda X: 3 + X if opposing else 4 - X),
        ]
    )
    analysis = ra.SystemForm(system, model2())
    analysis.run()
    if opposing:
        exact = 2 * norm.sf(3) if kind is ra.SeriesSystem else 0
    else:
        exact = norm.sf(3) if kind is ra.SeriesSystem else norm.sf(4)
    assert analysis.get_failure() == pytest.approx(exact, abs=1e-12)


def test_shared_components_nested_and_rescaling():
    a = ra.Component("a", lambda X: 2 - X)
    b = ra.Component("b", lambda X, Y: 3 - (X + Y) / np.sqrt(2))
    s = ra.SeriesSystem([a, ra.SeriesSystem([a, b])])
    baseline = ra.SystemForm(s, model2())
    baseline.run()
    scaled = ra.SystemForm(
        ra.SeriesSystem(
            [
                ra.Component("a", lambda X: 1000 * (2 - X)),
                ra.Component("b", lambda X, Y: 0.01 * (3 - (X + Y) / np.sqrt(2))),
            ]
        ),
        model2(),
    )
    scaled.run()
    assert len(baseline.component_results) == 2
    assert baseline.get_failure() == pytest.approx(scaled.get_failure(), rel=1e-5)
    assert baseline.correlation[0, 1] == pytest.approx(1 / np.sqrt(2))


def test_single_component_and_full_order_ddm():
    model = model2()
    options = ra.AnalysisOptions()
    options.set_diff_mode("ddm")

    def component(X):
        return 3 - X, np.array([[-1.0], [0.0]])

    analysis = ra.SystemForm(
        ra.SeriesSystem([ra.Component("a", component)]), model, options
    )
    analysis.run()
    assert analysis.get_failure() == pytest.approx(norm.sf(3))


def test_nonconvergence_invalidates_system_and_rerun():
    options = ra.AnalysisOptions()
    system = ra.SeriesSystem([ra.Component("a", lambda X: 3 - X)])
    analysis = ra.SystemForm(system, model2(), options)
    analysis.run()
    options.set_imax(1)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        with pytest.raises(RuntimeError, match="Component 'a'.*did not converge"):
            analysis.run()
    assert not analysis.results_valid
    with pytest.raises(ValueError, match="no valid result"):
        analysis.get_failure()


def test_zero_gradient_rejected():
    analysis = ra.SystemForm(
        ra.SeriesSystem([ra.Component("a", lambda X: X * 0 + 1)]), model2()
    )
    with pytest.raises(RuntimeError, match="nonzero finite gradient"):
        analysis.run()
    assert not analysis.results_valid


def test_mixed_topology_rejected():
    a = ra.Component("a", lambda X: X)
    with pytest.raises(TypeError, match="Mixed"):
        ra.SystemForm(ra.SeriesSystem([ra.ParallelSystem([a])]), model2())


@pytest.mark.parametrize(
    "probabilities,intersections",
    [
        ([0.1, 0.2], {}),
        ([0.1, 0.2], {(0, 1): 0.3}),
        ([float("nan")], {}),
        ([0.1, 0.2], {(0, 1): float("nan")}),
        ([0.8, 0.8], {(0, 1): 0.1}),
    ],
)
def test_invalid_bounds_inputs(probabilities, intersections):
    with pytest.raises(ValueError):
        ra.ditlevsen_bounds(probabilities, intersections)


def test_four_branch_against_nonlinear_reference():
    from scipy.integrate import quad

    s2 = np.sqrt(2)
    system = ra.SeriesSystem(
        [
            ra.Component("g1", lambda X, Y: 3 + 0.1 * (X - Y) ** 2 - (X + Y) / s2),
            ra.Component("g2", lambda X, Y: 3 + 0.1 * (X - Y) ** 2 + (X + Y) / s2),
            ra.Component("g3", lambda X, Y: X - Y + 6 / s2),
            ra.Component("g4", lambda X, Y: Y - X + 6 / s2),
        ]
    )
    analysis = ra.SystemForm(system, model2())
    analysis.run()
    # Tangent planes give |U1|>3 or |U2|>3 after an orthogonal rotation.
    p = norm.sf(3)
    assert analysis.get_failure() == pytest.approx(4 * p - 4 * p * p, rel=2e-4)
    reference = (
        2 * p + quad(lambda v: norm.pdf(v) * 2 * norm.sf(3 + 0.2 * v * v), -3, 3)[0]
    )
    rng = np.random.default_rng(2026)
    x, y = rng.standard_normal((2, 200000))
    estimate = system.failure_mask(X=x, Y=y).mean()
    se = np.sqrt(reference * (1 - reference) / len(x))
    assert abs(estimate - reference) < 5 * se
    # This nonlinear benchmark has genuine FORM model error.
    assert 1.1 < analysis.get_failure() / reference < 1.3


def test_original_cut_set_monte_carlo_shared_component():
    model = model2()
    model.add_variable(ra.Normal("Z", 0, 1))
    components = {
        "a": ra.Component("a", lambda X: 1 - X),
        "b": ra.Component("b", lambda Y: 1 - Y),
        "c": ra.Component("c", lambda Z: 1 - Z),
    }
    system = ra.CutSetSystem([["a", "b"], ["a", "c"]], components=components)
    options = ra.AnalysisOptions()
    options.set_samples(5000)
    options.target_cov = 0
    state = np.random.get_state()
    try:
        np.random.seed(2026)
        mc = ra.CrudeMonteCarlo(
            stochastic_model=model,
            limit_state=system.as_limit_state(),
            analysis_options=options,
        )
        mc.run()
    finally:
        np.random.set_state(state)
    p = norm.sf(1)
    exact = 2 * p * p - p**3
    assert abs(mc.get_failure() - exact) < 5 * np.sqrt(exact * (1 - exact) / mc.k)
    assert mc.cov_q_bar[mc.k - 1] > 0


def test_nataf_factorisation_invariance_with_constants():
    results = []
    for factor in ["cholesky", "svd"]:
        options = ra.AnalysisOptions()
        options.set_transform(factor)
        model = model2(0.4)
        model.add_variable(ra.Constant("C", 2))
        system = ra.SeriesSystem(
            [
                ra.Component("a", lambda X, C: C - X),
                ra.Component("b", lambda X, Y: 3 - (X + Y)),
            ]
        )
        analysis = ra.SystemForm(system, model, options)
        analysis.run()
        results.append(analysis)
    np.testing.assert_allclose(
        results[0].correlation, results[1].correlation, atol=1e-10
    )
    assert results[0].get_failure() == pytest.approx(results[1].get_failure(), rel=1e-7)


def test_unresolved_multivariate_zero_is_not_reported_as_safe(monkeypatch):
    model = model2(0.3)
    model.add_variable(ra.Normal("Z", 0, 1))
    model.set_correlation([[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]])
    system = ra.ParallelSystem(
        [
            ra.Component("a", lambda X: 3 - X),
            ra.Component("b", lambda Y: 3 - Y),
            ra.Component("c", lambda Z: 3 - Z),
        ]
    )
    monkeypatch.setattr(
        "pystra.system_form.multivariate_normal.cdf", lambda *a, **kw: 0.0
    )
    analysis = ra.SystemForm(system, model)
    with pytest.raises(RuntimeError, match="unresolved zero"):
        analysis.run()
    assert not analysis.results_valid


def test_single_rare_parallel_bounds_are_exact():
    result = ra.SystemForm(
        ra.ParallelSystem([ra.Component("a", lambda X: 8 - X)]), model2()
    )
    result.run()
    assert result.bounds == (result.get_failure(), result.get_failure())


@pytest.mark.parametrize(
    "kind,exact", [(ra.SeriesSystem, 0.75), (ra.ParallelSystem, 0.25)]
)
def test_three_correlated_normal_orthant(kind, exact):
    model = model2()
    model.add_variable(ra.Normal("Z", 0, 1))
    model.set_correlation([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    system = kind(
        [
            ra.Component("a", lambda X: -X),
            ra.Component("b", lambda Y: -Y),
            ra.Component("c", lambda Z: -Z),
        ]
    )
    result = ra.SystemForm(system, model)
    result.run()
    # Trivariate zero-threshold orthant: 1/8 + sum(asin(rho_ij))/(4*pi).
    assert result.get_failure() == pytest.approx(exact, abs=2e-6)


def test_three_identical_directions():
    system = ra.ParallelSystem(
        [
            ra.Component("a", lambda X: 2 - X),
            ra.Component("b", lambda X: 3 - X),
            ra.Component("c", lambda X: 4 - X),
        ]
    )
    result = ra.SystemForm(system, model2())
    result.run()
    assert result.get_failure() == pytest.approx(norm.sf(4), rel=1e-8, abs=0)
