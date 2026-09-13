"""Transform accuracy for very small tail probabilities.

A marginal transformation x = F^-1(Phi(u)) formed naively rounds Phi(u) to
one above u = 8.3, which sends x to infinity. Each tail is evaluated on the
side where its probability is small, with log probabilities where these
underflow; these tests check that against closed forms and references.
"""

import math

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

import pystra as ra

U_TAIL = np.array(
    [
        -37.0,
        -30.0,
        -20.0,
        -10.0,
        -6.0,
        -3.0,
        -0.5,
        0.0,
        0.5,
        3.0,
        6.0,
        10.0,
        20.0,
        30.0,
        37.0,
    ]
)


def unbounded_tails():
    return {
        "Normal": ra.Normal("X", 10, 3),
        "Lognormal": ra.Lognormal("X", 10, 3),
        "ShiftedLognormal": ra.ShiftedLognormal("X", 10, 3, 2.0),
        "Gumbel": ra.Gumbel("X", 10, 3),
        "GumbelMin": ra.GumbelMin("X", 10, 3),
        "Weibull": ra.Weibull("X", 10, 3),
        "Gamma": ra.Gamma("X", 10, 3),
        "Frechet": ra.Frechet("X", 10, 3),
        "ChiSquare": ra.ChiSquare("X", 8, 4),
        "GEV": ra.GEV("X", 10, 3, shape=0.2),
        "GEVMin": ra.GEVMin("X", 10, 3, shape=0.2),
        "Maximum": ra.Maximum("Q", ra.Gumbel("X", 10, 3), N=50),
        "Maximum of lognormal": ra.Maximum("Q", ra.Lognormal("X", 10, 3), N=5),
        "MaxParent": ra.MaxParent("Q", ra.Gumbel("X", 10, 3), N=50),
        "MaxParent of normal": ra.MaxParent("Q", ra.Normal("X", 10, 3), N=1000),
    }


@pytest.mark.parametrize("name", list(unbounded_tails()))
def test_transform_round_trips_far_into_both_tails(name):
    distribution = unbounded_tails()[name]
    x = np.asarray(distribution.u_to_x(U_TAIL))
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)
    np.testing.assert_allclose(distribution.x_to_u(x), U_TAIL, rtol=1e-12, atol=1e-9)
    assert float(distribution.u_to_x(U_TAIL[-1])) == x[-1]


@pytest.mark.parametrize(
    "distribution",
    [
        ra.Normal("X", 10, 3),
        ra.Lognormal("X", 10, 3),
        ra.Gumbel("X", 10, 3),
        ra.GumbelMin("X", 10, 3),
        ra.Maximum("Q", ra.Gumbel("X", 10, 3), N=50),
        ra.MaxParent("Q", ra.Gumbel("X", 10, 3), N=50),
    ],
)
def test_closed_forms_extend_beyond_double_probabilities(distribution):
    u = np.array([-60.0, -40.0, 40.0, 60.0])
    x = np.asarray(distribution.u_to_x(u))
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)
    np.testing.assert_allclose(distribution.x_to_u(x), u, rtol=1e-12)


def test_gumbel_tails_match_high_precision_reference():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    gumbel = ra.Gumbel("X", 10, 3)
    loc, scale = gumbel.dist_obj.kwds["loc"], gumbel.dist_obj.kwds["scale"]
    for u in (-37.0, -9.0, 9.0, 20.0, 37.0):
        log_f = mp.log1p(-mp.ncdf(-u)) if u > 0 else mp.log(mp.ncdf(u))
        reference = float(loc - scale * mp.log(-log_f))
        assert float(gumbel.u_to_x(u)) == pytest.approx(reference, rel=4e-15)


@pytest.mark.parametrize("N", [5, 1000])
def test_maxima_of_a_gumbel_are_shifted_gumbels(N):
    parent = ra.Gumbel("X", 10, 3)
    loc, scale = parent.dist_obj.kwds["loc"], parent.dist_obj.kwds["scale"]
    maximum = ra.Maximum("Q", parent, N=N)
    maximum_parent = ra.MaxParent("P", parent, N=N)
    up = ra.Gumbel("U", loc=loc + scale * np.log(N), scale=scale)
    down = ra.Gumbel("D", loc=loc - scale * np.log(N), scale=scale)
    np.testing.assert_allclose(maximum.u_to_x(U_TAIL), up.u_to_x(U_TAIL), rtol=1e-12)
    np.testing.assert_allclose(
        maximum_parent.u_to_x(U_TAIL), down.u_to_x(U_TAIL), rtol=1e-12
    )
    x = np.array([-40.0, 0.0, 30.0, 200.0])
    np.testing.assert_allclose(maximum.logsf(x), up.logsf(x), rtol=1e-12)
    np.testing.assert_allclose(maximum_parent.logcdf(x), down.logcdf(x), rtol=1e-12)


def test_jacobian_uses_log_densities_where_densities_underflow():
    gumbel = ra.Gumbel("X", 10, 3)
    for u in (-37.6, 37.6):
        x = float(gumbel.u_to_x(u))
        jacobian = gumbel.jacobian(np.array([u]), np.array([x]))[0, 0]
        h = 1e-6 * abs(x)
        numeric = (gumbel.x_to_u(x + h) - gumbel.x_to_u(x - h)) / (2 * h)
        assert np.isfinite(jacobian) and jacobian == pytest.approx(numeric, rel=1e-6)


class _Logistic(ra.Distribution):
    """A SciPy distribution with an exact log-CDF but no log-quantile."""

    def __init__(self):
        super().__init__(name="L", dist_obj=stats.logistic())


def test_generic_solver_inverts_log_probabilities_beyond_underflow():
    logistic = _Logistic()
    u = np.array([-45.0, -39.0])
    x = np.asarray(logistic.u_to_x(u))
    # log F(x) = -log1p(exp(-x)), which equals x to double precision here
    np.testing.assert_allclose(x, log_ndtr(u), rtol=1e-12)
    np.testing.assert_allclose(logistic.x_to_u(x), u, rtol=1e-12)


def test_tail_inverse_is_checked_against_the_cdf():
    # SciPy's beta.ppf stalls near 4.1e-50 for probabilities below about 1e-100
    beta = ra.Beta("X", q=2, r=5, lower=0, upper=10)
    for u in (-30.0, -37.0):
        x = float(beta.u_to_x(u))
        assert float(beta.cdf(x)) == pytest.approx(ndtr(u), rel=1e-10)


@pytest.mark.parametrize("shape", [-0.2, 0.0, 0.2])
def test_gevmin_is_the_reflected_gev(shape):
    by_moments = ra.GEVMin("X", 10, 3, shape=shape)
    p = (np.arange(200_000) + 0.5) / 200_000
    x = by_moments.ppf(p)
    assert x.mean() == pytest.approx(10, abs=2e-3)
    assert x.std() == pytest.approx(3, rel=5e-3)
    minimum = ra.GEVMin("X", shape=shape, loc=12.0, scale=2.5)
    maximum = ra.GEV("Y", shape=shape, loc=-12.0, scale=2.5)
    assert minimum.mean == pytest.approx(-maximum.mean)
    points = np.array([-5.0, 8.0, 12.0, 15.0])
    np.testing.assert_allclose(minimum.cdf(points), maximum.sf(-points), rtol=1e-12)
    u = np.array([-8.0, -1.0, 0.0, 2.0, 9.0])
    np.testing.assert_allclose(minimum.u_to_x(u), -maximum.u_to_x(-u), rtol=1e-12)


def test_max_parent_of_lognormal_inverts_for_large_n():
    distribution = ra.MaxParent("Q", ra.Lognormal("X", 10, 3), N=1000)
    p = np.array([1e-10, 0.01, 0.5, 0.99])
    np.testing.assert_allclose(distribution.cdf(distribution.ppf(p)), p, rtol=1e-9)
    assert np.isfinite(distribution.mean) and distribution.std > 0


def test_uniform_upper_tail_is_measured_from_the_upper_bound():
    uniform = ra.Uniform("X", lower=2.0, upper=5.0)
    x = float(uniform.u_to_x(7.0))
    assert 5.0 - x == pytest.approx(3.0 * ndtr(-7.0), rel=1e-12)


def test_zero_inflated_upper_tail_uses_the_parent():
    distribution = ra.ZeroInflated("Z", ra.Lognormal("X", 10, 3), p=0.3)
    u = np.array([3.0, 10.0, 30.0])
    x = np.asarray(distribution.u_to_x(u))
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)
    np.testing.assert_allclose(distribution.x_to_u(x), u, rtol=1e-12)
    assert distribution.u_to_x(-3.0) == 0.0


def test_student_t_copula_uses_the_marginal_tail_functions():
    marginals = [ra.Maximum("Q", ra.Gumbel("X", 10, 3), N=5), ra.Gumbel("Y", 10, 3)]
    joint = ra.JointDistribution(marginals, ra.StudentTCopula(np.eye(2), 4))
    transform = joint.make_transformation("rosenblatt")
    u = np.array([9.5, -9.5])
    x = transform.u_to_x(u)
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(transform.x_to_u(x), u, rtol=1e-9)


@pytest.mark.parametrize("resistance_mean", [22.0, 34.0, 60.0])
def test_sampling_with_a_gumbel_load_matches_the_exact_integral(resistance_mean):
    # Line sampling scans far past u = 8.3, where the naive transform fails
    resistance = ra.Lognormal("R", resistance_mean, 2.0)
    load = ra.Gumbel("S", 8.0, 1.5)
    model = ra.StochasticModel()
    model.add_variable(resistance)
    model.add_variable(load)
    limit_state = ra.LimitState(lambda R, S: R - S)
    pf, _ = integrate.quad(
        lambda r: np.exp(resistance.logpdf(r) + load.logsf(r)),
        resistance.ppf(1e-12),
        resistance.isf(1e-15),
        points=[resistance_mean],
        epsabs=0,
        epsrel=1e-10,
        limit=500,
    )
    beta = -norm.ppf(pf)
    form = ra.FORM(model, limit_state)
    assert form.run().beta == pytest.approx(beta, abs=0.01)
    options = ra.SimulationOptions(n_samples=50)
    line = ra.LineSampling(model, limit_state, form=form, options=options, rng=1).run()
    assert line.beta == pytest.approx(beta, abs=2e-3)
    options = ra.SimulationOptions(n_samples=4000)
    importance = ra.ImportanceSampling(
        model, limit_state, form=form, options=options, rng=1
    ).run()
    assert importance.beta == pytest.approx(beta, abs=0.05)


class _ScalarGumbel(ra.Distribution):
    """A subclass whose overrides accept only scalars, as 1.x examples did."""

    def __init__(self, name):
        super().__init__(name=name, dist_obj=stats.gumbel_r(loc=8.0, scale=1.5))

    def cdf(self, x):
        return math.exp(-math.exp(-(x - 8.0) / 1.5))

    def ppf(self, p):
        return 8.0 - 1.5 * math.log(-math.log(p))


def test_scalar_only_subclass_methods_still_work():
    distribution = _ScalarGumbel("S")
    for u in (-2.0, 0.5, 2.5):
        x = float(distribution.u_to_x(u))
        assert float(distribution.x_to_u(x)) == pytest.approx(u, abs=1e-9)
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("R", 20.0, 2.0))
    model.add_variable(distribution)
    result = ra.FORM(model, ra.LimitState(lambda R, S: R - S)).run()
    assert result.converged


def test_zero_inflated_lower_tail_stays_in_the_atom():
    distribution = ra.ZeroInflated("Z", ra.Lognormal("X", 10, 3), p=0.3)
    for u in (-6.0, -10.0, -30.0):
        assert distribution.u_to_x(u) == 0.0


@pytest.mark.parametrize(
    "distribution",
    [ra.Lognormal("L", 10, 3), ra.ShiftedLognormal("S", 10, 3, 2.0)],
)
def test_lognormal_functions_propagate_nan(distribution):
    for function in (
        distribution.cdf,
        distribution.sf,
        distribution.pdf,
        distribution.logpdf,
        distribution.x_to_u,
    ):
        assert np.isnan(function(np.nan))
    lower = distribution._shift
    assert distribution.cdf(lower) == 0.0 and distribution.pdf(lower - 1) == 0.0
