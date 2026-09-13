"""Regressions from the September 2026 runtime and numerical review."""

import numpy as np
import pytest
from scipy.special import log_ndtr, ndtri
from scipy.stats import norm

import pystra as ra


def problem():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    model.add_variable(ra.Normal("Y", 0, 1))
    return model, ra.LimitState(lambda X, Y: 3 - X - Y)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("partial", [False, True])
def test_monte_carlo_rejects_invalid_samples(bad, partial):
    model, _ = problem()

    def evaluate(X, Y):
        values = np.full_like(X, bad)
        if partial:
            values[0] = -1
        return values

    analysis = ra.CrudeMonteCarlo(
        model,
        ra.LimitState(evaluate),
        options=ra.SimulationOptions(n_samples=10),
        rng=7,
    )
    with pytest.raises(ra.AnalysisError, match="Nonfinite limit-state values"):
        analysis.run()
    assert not analysis._results_valid and analysis._Pf is None


@pytest.mark.parametrize(
    "method", [ra.CrudeMonteCarlo, ra.LineSampling, ra.ImportanceSampling]
)
def test_external_solver_failure_clears_success_and_preserves_cause(method):
    model, _ = problem()
    failing = False
    error = RuntimeError("external solver failed")

    def evaluate(X, Y):
        if failing:
            raise error
        return 3 - X - Y

    limit_state = ra.LimitState(evaluate)
    form = ra.FORM(model, limit_state)
    form.run()
    kwargs = {} if method is ra.CrudeMonteCarlo else {"form": form}
    analysis = method(
        model, limit_state, options=ra.SimulationOptions(n_samples=4), rng=3, **kwargs
    )
    analysis.run()
    failing = True
    with pytest.raises(ra.AnalysisError, match="external solver failed") as info:
        analysis.run()
    cause = info.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert cause is error
    assert not analysis._results_valid and analysis._Pf is None


def test_line_sampling_requires_successful_root_refinement(monkeypatch):
    model, limit_state = problem()
    form = ra.FORM(model, limit_state)
    form.run()

    def fail(*args, **kwargs):
        raise RuntimeError("root iteration limit")

    monkeypatch.setattr("pystra.reliability.line_sampling.optimize.brentq", fail)
    analysis = ra.LineSampling(
        model, limit_state, form=form, options=ra.SimulationOptions(n_samples=2), rng=4
    )
    with pytest.raises(ra.AnalysisError, match="Line 1.*root iteration limit"):
        analysis.run()
    assert not analysis._results_valid


def test_explicit_correlation_cannot_be_reset_by_adding_a_variable():
    model, _ = problem()
    limit_state = ra.LimitState(lambda X, Y: 3 + X - Y)
    model.set_correlation([[1, 0.8], [0.8, 1]])
    before = ra.FORM(model, limit_state).run()
    with pytest.raises(ra.ModelError, match="before setting correlation"):
        model.add_variable(ra.Normal("Z", 0, 1))
    assert model.get_names() == ["X", "Y"]
    assert model.n_marg == 2
    assert ra.FORM(model, limit_state).run().beta == before.beta
    assert before.beta == pytest.approx(3 / np.sqrt(0.4))
    model.add_variable(ra.Constant("c", 1))
    assert model.correlation[0, 1] == 0.8


def test_wrong_correlation_dimensions_leave_existing_dependence_intact():
    model, _ = problem()
    model.set_copula(ra.GaussianCopula([[1, 0.4], [0.4, 1]]))
    copula = model.copula
    with pytest.raises(ra.ModelError, match="dimensions"):
        model.set_correlation(np.eye(3))
    assert model.copula is copula


@pytest.mark.parametrize("fit", ["curve", "point"])
def test_sorm_recomputes_internal_form_after_changed_limit_state(fit):
    model, limit_state = problem()
    analysis = ra.SORM(model, limit_state, options=ra.SORMOptions(fit=fit))
    old = analysis.run()
    analysis.limit_state = ra.LimitState(lambda X, Y: 1 - X - Y)
    result = analysis.run()
    assert old.beta == pytest.approx(3 / np.sqrt(2))
    assert result.beta == pytest.approx(1 / np.sqrt(2))


@pytest.mark.parametrize("method", [ra.SORM, ra.LineSampling, ra.ImportanceSampling])
def test_supplied_form_detects_model_edits_and_can_be_refreshed(method):
    model, limit_state = problem()
    form = ra.FORM(model, limit_state)
    form.run()
    kwargs = (
        {}
        if method is ra.SORM
        else {"options": ra.SimulationOptions(n_samples=4), "rng": 4}
    )
    analysis = method(model, limit_state, form=form, **kwargs)
    model.variable("X").set_location(1)
    with pytest.raises(ra.AnalysisError, match="model inputs have changed"):
        analysis.run()
    form.run()
    result = analysis.run()
    nested = result.form if method is ra.SORM else result.diagnostics["form"]
    assert nested.beta == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("method", [ra.LineSampling, ra.ImportanceSampling])
def test_simulations_recompute_internal_form_on_rerun(method):
    model, limit_state = problem()
    analysis = method(
        model, limit_state, options=ra.SimulationOptions(n_samples=4), rng=4
    )
    first = analysis.run()
    limit_state.expression = lambda X, Y: 1 - X - Y
    second = analysis.run()
    assert first.diagnostics["form"].beta == pytest.approx(3 / np.sqrt(2))
    assert second.diagnostics["form"].beta == pytest.approx(1 / np.sqrt(2))


@pytest.mark.parametrize("method", [ra.SORM, ra.LineSampling, ra.ImportanceSampling])
def test_supplied_form_rejects_another_limit_state(method):
    model, limit_state = problem()
    form = ra.FORM(model, limit_state)
    form.run()
    with pytest.raises(ra.ModelError, match="same model and limit-state expression"):
        method(model, ra.LimitState(lambda X, Y: 1 - X - Y), form=form).run()


@pytest.mark.parametrize("method", [ra.LineSampling, ra.ImportanceSampling])
def test_supplied_form_requires_matching_coordinates(method):
    model, limit_state = problem()
    model.set_correlation([[1, 0.4], [0.4, 1]])
    form = ra.FORM(model, limit_state, options=ra.FORMOptions(transform="svd"))
    form.run()
    with pytest.raises(ra.ModelError, match="same transformation"):
        method(model, limit_state, form=form).run()


@pytest.mark.parametrize("fit", ["curve", "point"])
@pytest.mark.parametrize("weights", [(1, 0), (0, 1), (-1, 0), (1, 1e-12)])
def test_sorm_handles_axis_aligned_and_near_axis_aligned_surfaces(fit, weights):
    model, _ = problem()
    a, b = weights
    limit_state = ra.LimitState(lambda X, Y: 3 - a * X - b * Y)
    result = ra.SORM(model, limit_state, options=ra.SORMOptions(fit=fit)).run()
    beta = 3 / np.hypot(a, b)
    assert result.beta == pytest.approx(beta, abs=1e-6)
    assert result.failure_probability == pytest.approx(norm.sf(beta), rel=1e-5)
    np.testing.assert_allclose(result.curvatures, 0, atol=1e-7)


@pytest.mark.parametrize("method", [ra.CrudeMonteCarlo, ra.ImportanceSampling])
def test_fixed_budget_is_completed_without_claiming_precision(method):
    model, limit_state = problem()
    result = method(
        model,
        limit_state,
        options=ra.SimulationOptions(n_samples=100, target_cov=0),
        rng=4,
    ).run()
    assert result.status == "completed" and result.n_samples == 100
    assert result.coefficient_of_variation > 0
    assert "budget completed" in result.message


def test_normal_extreme_quantiles_and_lower_cdf_remain_finite():
    distribution = ra.Normal("X", 0, 1)
    probabilities = np.array(
        [1e-300, 1e-100, 1e-20, 1e-10, 0.5, np.nextafter(1.0, 0.0)]
    )
    np.testing.assert_allclose(distribution.ppf(probabilities), ndtri(probabilities))
    np.testing.assert_allclose(
        distribution.cdf(distribution.ppf(probabilities)),
        probabilities,
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize("kind", [ra.Maximum, ra.MaxParent])
def test_compound_moments_are_deterministic_and_match_gumbel_identity(
    kind, monkeypatch
):
    def random_forbidden(*args, **kwargs):
        raise AssertionError("Distribution construction must not draw random samples")

    monkeypatch.setattr(np.random, "random", random_forbidden)
    parent = ra.Gumbel("G", loc=5, scale=2)
    distribution = kind("Q", parent, N=5)
    shift = 2 * np.log(5) * (1 if kind is ra.Maximum else -1)
    assert distribution.mean == pytest.approx(parent.mean + shift, abs=2e-7)
    assert distribution.std == pytest.approx(parent.std, abs=2e-7)
    second = kind("Q", parent, N=5)
    assert (distribution.mean, distribution.std) == (second.mean, second.std)


def test_maximum_normal_pair_matches_analytic_moments():
    maximum = ra.Maximum("Q", ra.Normal("X", 0, 1), N=2)
    assert maximum.mean == pytest.approx(1 / np.sqrt(np.pi), abs=1e-8)
    assert maximum.std == pytest.approx(np.sqrt(1 - 1 / np.pi), abs=1e-8)


def test_normal_max_parent_has_finite_moments_and_log_tail_inverse():
    distribution = ra.MaxParent("Q", ra.Normal("X", 0, 1), N=5)
    assert np.isfinite(distribution.mean) and distribution.std > 0
    probabilities = np.array([1e-100, 1e-20, 0.001, 0.5])
    np.testing.assert_allclose(
        log_ndtr(distribution.ppf(probabilities)), 5 * np.log(probabilities), rtol=1e-12
    )


@pytest.mark.parametrize("differentiation", ["ffd", "ddm"])
def test_gradient_batch_preserves_point_order(differentiation):
    model, _ = problem()

    def evaluate(X, Y):
        return X + 2 * Y, np.array([1.0, 2.0])

    points = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    values, gradient = ra.LimitState(evaluate)._evaluate_lsf(
        points, model, differentiation=differentiation
    )
    np.testing.assert_allclose(np.ravel(values), [9, 12, 15])
    np.testing.assert_allclose(gradient, [[1, 1, 1], [2, 2, 2]], atol=1e-10)
