"""Method-layer regressions for rare-event numerics and runtime contracts."""

import numpy as np
import pytest
from scipy.stats import norm

import pystra as ra


def _problem(beta=3):
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    model.add_variable(ra.Normal("Y", 0, 1))
    return model, ra.LimitState(lambda X, Y: beta - X)


@pytest.mark.parametrize("method", [ra.SORM, ra.LineSampling, ra.ImportanceSampling])
@pytest.mark.parametrize("initial_form", [False, True])
def test_form_assignment_after_construction_is_used(method, initial_form):
    model, limit_state = _problem()
    original = ra.FORM(model, limit_state)
    original.run()
    kwargs = (
        {}
        if method is ra.SORM
        else {"options": ra.SimulationOptions(n_samples=4), "rng": 1}
    )
    analysis = method(
        model, limit_state, form=original if initial_form else None, **kwargs
    )
    limit_state.expression = lambda X, Y: 2 - X
    replacement = ra.FORM(model, limit_state)
    replacement.run()
    analysis.form = replacement
    result = analysis.run()
    assert analysis.form is replacement
    nested = result.form if method is ra.SORM else result.diagnostics["form"]
    assert nested.beta == pytest.approx(2)
    analysis.form = None
    limit_state.expression = lambda X, Y: 4 - X
    result = analysis.run()
    assert analysis.form is not replacement
    nested = result.form if method is ra.SORM else result.diagnostics["form"]
    assert nested.beta == pytest.approx(4)
    with pytest.raises(TypeError, match="FORM analysis"):
        analysis.form = result


@pytest.mark.parametrize("differentiation", ["no", "ffd", "ddm"])
def test_wrong_argument_names_are_model_errors(differentiation):
    model, _ = _problem()
    with pytest.raises(ra.ModelError, match="signature"):
        ra.LimitState(lambda X, Z: X + Z)._evaluate_lsf(
            np.zeros((2, 1)), model, differentiation=differentiation
        )


def test_type_error_inside_evaluator_remains_analysis_error():
    model, _ = _problem()
    original = TypeError("solver implementation failed")

    def evaluate(X, Y):
        raise original

    with pytest.raises(ra.AnalysisError) as info:
        ra.LimitState(evaluate)._evaluate_lsf(np.zeros((2, 1)), model)
    assert info.value.__cause__ is original


@pytest.mark.parametrize("beta", [12.0, 38.0, 40.0])
@pytest.mark.parametrize("fit", ["curve", "point"])
def test_sorm_preserves_index_after_probability_underflow(beta, fit):
    from scipy.special import log_ndtr, ndtri_exp

    model, limit_state = _problem()
    analysis = ra.SORM(model, limit_state)
    kappa = np.array([0.01])
    mills = np.exp(norm.logpdf(beta) - log_ndtr(-beta))
    if fit == "curve":
        analysis._pf_breitung(beta, kappa)
        analysis._pf_breitung_m(beta, kappa)
    else:
        analysis._pf_breitung_pf(beta, kappa, kappa)
        analysis._pf_breitung_m_pf(beta, kappa, kappa)
    for suffix, factor in [("", beta), ("_m", mills)]:
        log_pf = log_ndtr(-beta) - 0.5 * np.log1p(factor * 0.01)
        assert getattr(analysis, "_betag_breitung" + suffix) == pytest.approx(
            -ndtri_exp(log_pf)
        )
        np.testing.assert_allclose(
            getattr(analysis, "_pf2_breitung" + suffix),
            np.exp(log_pf),
            rtol=1e-12,
            atol=0,
        )


@pytest.mark.parametrize("beta", [28, 40])
def test_importance_sampling_tiny_probability_has_relative_uncertainty(beta):
    model, limit_state = _problem(beta)
    result = ra.ImportanceSampling(
        model,
        limit_state,
        options=ra.SimulationOptions(n_samples=4000, target_cov=0),
        rng=1,
    ).run()
    assert (result.failure_probability > 0) == (beta == 28)
    assert abs(result.beta - beta) < 0.03
    assert (
        result.diagnostics["history"]["coefficient_of_variation"][-1]
        == result.coefficient_of_variation
    )
    assert 0.01 < result.coefficient_of_variation < 0.2


def test_importance_weights_include_log_determinant_and_mask_safe_points():
    model, limit_state = _problem()
    analysis = ra.CrudeMonteCarlo(
        model, limit_state, options=ra.SimulationOptions(sampling_std=2)
    )
    analysis._nrv = 1100
    analysis._initialize_variables()
    analysis._block_size = 2
    analysis.point = np.zeros((1100, 1))
    analysis._u = np.full((1100, 2), np.sqrt(2 * np.log(2) / 0.75))
    analysis._I = np.array([1, 0])
    with np.errstate(over="raise", invalid="raise"):
        analysis._compute_sum_update()
    np.testing.assert_allclose(analysis._q, [1, 0], rtol=1e-10)


def test_line_scan_caps_both_independent_and_correlated_coordinates(monkeypatch):
    model, limit_state = _problem()
    analysis = ra.LineSampling(model, limit_state)
    analysis.init_run()
    alpha = np.array([0.6, 0.8])
    v = np.array([-0.8, 0.6])
    analysis.transform.inv_T = np.array([[1.0, 0.0], [0.8, 0.6]])
    visited = []

    def evaluate(c, v, alpha, marg):
        u = v + c * alpha
        visited.extend([*u, *(analysis.transform.inv_T @ u)])
        return 20 - c

    monkeypatch.setattr(analysis, "_eval_g_at_c", evaluate)
    assert analysis._find_line_intersection(v, alpha, 20, []) == pytest.approx(20)
    assert max(np.abs(visited)) <= 37


def test_line_scan_reports_unresolved_crossing_beyond_cap():
    model, limit_state = _problem(40)
    with pytest.raises(ra.AnalysisError, match="supported normal range"):
        ra.LineSampling(
            model, limit_state, options=ra.SimulationOptions(n_samples=2), rng=1
        ).run()


def test_line_probability_dispersion_does_not_square_tiny_probabilities():
    model, _ = _problem()
    limit_state = ra.LimitState(lambda X, Y: 28 + 0.02 * Y**2 - X)
    result = ra.LineSampling(
        model, limit_state, options=ra.SimulationOptions(n_samples=30), rng=1
    ).run()
    assert result.failure_probability > 0
    assert 0 < result.coefficient_of_variation < 1


@pytest.mark.parametrize("kind", [ra.Maximum, ra.MaxParent])
@pytest.mark.parametrize(
    "parent_type",
    [
        ra.Normal,
        ra.Lognormal,
        ra.Gumbel,
        ra.GumbelMin,
        ra.Gamma,
        ra.Weibull,
        ra.Frechet,
        ra.Uniform,
        ra.ShiftedExponential,
        ra.ShiftedRayleigh,
        ra.ChiSquare,
    ],
)
@pytest.mark.parametrize("exponent", [1.5, 5, 50, 1000])
@pytest.mark.filterwarnings(
    "ignore:.*moment quadrature did not reach tolerance:RuntimeWarning"
)
def test_compound_moment_probe_cases_construct(kind, parent_type, exponent):
    parent = (
        parent_type("P", 8, 4)
        if parent_type is ra.ChiSquare
        else parent_type("P", 10, 3)
    )
    distribution = kind("Q", parent, exponent)
    assert np.isfinite(distribution.mean)
    assert np.isfinite(distribution.std) and distribution.std > 0


@pytest.mark.parametrize("kind", [ra.Maximum, ra.MaxParent])
def test_frechet_power_moments_remain_exact_near_infinite_variance(kind):
    parent = ra.Frechet("P", scale=3, shape=2.001)
    distribution = kind("Q", parent, 1000)
    factor = (1000 if kind is ra.Maximum else 0.001) ** (1 / 2.001)
    assert distribution.mean == pytest.approx(factor * parent.mean, rel=1e-14)
    assert distribution.std == pytest.approx(factor * parent.std, rel=1e-14)


@pytest.mark.parametrize("kind", [ra.Maximum, ra.MaxParent])
@pytest.mark.filterwarnings(
    "ignore:.*moment quadrature did not reach tolerance:RuntimeWarning"
)
def test_exponential_power_moments_match_polygamma_identity(kind):
    from scipy.special import digamma, polygamma

    parent = ra.ShiftedExponential("P", 10, 3)
    distribution = kind("Q", parent, 1000)
    power = 1000 if kind is ra.Maximum else 0.001
    mean = 7 + 3 * (digamma(1 + power) - digamma(1))
    variance = 9 * (polygamma(1, 1) - polygamma(1, 1 + power))
    assert distribution.mean == pytest.approx(mean, abs=2e-7)
    assert distribution.std == pytest.approx(np.sqrt(variance), rel=2e-6)


def test_unmet_moment_tolerance_warns_and_returns_finite_estimate():
    from pystra.distributions._moments import _quantile_moments
    from types import SimpleNamespace

    parent = ra.Normal("P", 0, 1)
    # A quantile jump between quadrature nodes deliberately defeats smooth
    # Gaussian quadrature, without making either of the moments infinite.
    distribution = SimpleNamespace(
        parent=parent, N=2, _ppf_log=lambda lp: (lp > -1.5).astype(float)
    )
    with pytest.warns(RuntimeWarning, match="finest finite estimate"):
        mean, std = _quantile_moments(distribution, 0, 1)
    assert 0 < mean < 1 and 0 < std <= 0.5


def _evaluate_with_module_default(X, Y, numeric=np):
    return 3 - numeric.asarray(X) - numeric.asarray(Y)


def test_limit_state_still_pickles_after_evaluation():
    import pickle

    model, _ = _problem()
    limit_state = ra.LimitState(_evaluate_with_module_default)
    limit_state._evaluate_lsf(np.zeros((2, 1)), model)
    clone = pickle.loads(pickle.dumps(limit_state))
    values, _ = clone._evaluate_lsf(np.zeros((2, 1)), model)
    assert np.ravel(values)[0] == 3


@pytest.mark.parametrize("fit", ["curve", "point"])
@pytest.mark.parametrize("formula", ["breitung", "modified_breitung"])
def test_sorm_result_reports_finite_index_after_underflow(fit, formula):
    model, _ = _problem()
    limit_state = ra.LimitState(lambda X, Y: 40 + 0.005 * Y**2 - X)
    options = ra.SORMOptions(fit=fit, formula=formula)
    result = ra.SORM(model, limit_state, options=options).run()
    assert result.failure_probability == 0.0
    assert result.beta == pytest.approx(40.0042, abs=2e-4)
