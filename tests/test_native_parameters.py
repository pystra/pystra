"""Distributions built from native parameters instead of mean and std."""

import numpy as np
import pytest
from scipy import stats
from scipy.special import gamma

from pystra.distributions import (
    GEV,
    Beta,
    ChiSquare,
    Frechet,
    Gamma,
    GEVMin,
    Gumbel,
    GumbelMin,
    Lognormal,
    Normal,
    ShiftedExponential,
    ShiftedRayleigh,
    Uniform,
    Weibull,
)

NATIVE = [
    (Gumbel, {"loc": 8.9, "scale": 1.56}, stats.gumbel_r(loc=8.9, scale=1.56)),
    (GumbelMin, {"loc": 12, "scale": 0.8}, stats.gumbel_l(loc=12, scale=0.8)),
    (
        GEV,
        {"shape": 0.1, "loc": 10, "scale": 2},
        stats.genextreme(c=-0.1, loc=10, scale=2),
    ),
    (Frechet, {"scale": 10, "shape": 4}, stats.invweibull(c=4, scale=10)),
    (
        Lognormal,
        {"log_mean": 2.3, "log_std": 0.1},
        stats.lognorm(s=0.1, scale=np.exp(2.3)),
    ),
    (Uniform, {"lower": 4, "upper": 6}, stats.uniform(loc=4, scale=2)),
    (
        Weibull,
        {"scale": 10, "shape": 2.5, "lower": 1},
        stats.weibull_min(c=2.5, loc=1, scale=10),
    ),
    (Beta, {"q": 2, "r": 5, "lower": 1, "upper": 3}, stats.beta(2, 5, loc=1, scale=2)),
    (Gamma, {"rate": 0.5, "shape": 4}, stats.gamma(a=4, scale=2)),
    (ShiftedExponential, {"rate": 2, "shift": 1}, stats.expon(loc=1, scale=0.5)),
    (ShiftedRayleigh, {"scale": 2, "shift": 1}, stats.rayleigh(loc=1, scale=2)),
    (ChiSquare, {"df": 3}, stats.chi2(df=3)),
]


@pytest.mark.parametrize(
    "cls,params,reference", NATIVE, ids=[cls.__name__ for cls, _, _ in NATIVE]
)
def test_native_parameters_match_scipy(cls, params, reference):
    dist = cls("X", **params)
    assert dist.mean == pytest.approx(reference.mean(), rel=1e-9)
    assert dist.std == pytest.approx(reference.std(), rel=1e-9)
    probabilities = [0.1, 0.5, 0.9]
    x = reference.ppf(probabilities)
    np.testing.assert_allclose(
        [float(dist.cdf(value)) for value in x], probabilities, rtol=1e-9
    )


def test_gev_min_native_parameters_reproduce_moments():
    shape = 0.1
    g1, g2 = gamma(1 - shape), gamma(1 - 2 * shape)
    scale = 20 * shape / np.sqrt(g2 - g1**2)
    # Minima: mean = loc - scale (g1 - 1) / shape, the reflection of the GEV
    loc = 100 + scale / shape * (g1 - 1)
    by_moments = GEVMin("X", 100, 20, shape)
    native = GEVMin("X", shape=shape, loc=loc, scale=scale)
    assert native.mean == pytest.approx(100)
    assert native.std == pytest.approx(20)
    reflected = stats.genextreme(c=-shape, loc=-loc, scale=scale)
    assert native.mean == pytest.approx(-reflected.mean())
    for x in (80.0, 100.0, 120.0):
        assert float(native.cdf(x)) == pytest.approx(float(by_moments.cdf(x)))


def test_either_moments_or_native_parameters():
    with pytest.raises(TypeError, match="mean and std, or loc and scale, not both"):
        Gumbel("X", 10, 2, loc=1, scale=1)
    with pytest.raises(TypeError, match="needs mean and std, or loc and scale"):
        Gumbel("X", loc=1)
    with pytest.raises(TypeError, match="needs mean and std, or df"):
        ChiSquare("X", 10)
    with pytest.raises(TypeError, match="needs shape"):
        GEV("X", 10, 2)


def test_input_type_is_removed():
    with pytest.raises(TypeError):
        Gumbel("X", 10, 2, input_type="par")


def test_start_point_is_keyword_only():
    with pytest.raises(TypeError):
        Normal("X", 0, 1, 0.5)
    assert Normal("X", 0, 1, start_point=0.5).start_point == 0.5


def test_bounds_are_named_keywords():
    weibull = Weibull("W", 10, 3, lower=2)
    beta = Beta("B", 5, 1, lower=3, upper=10)
    assert weibull.parameters["lower"] == 2
    assert beta.parameters["lower"] == 3
    assert beta.parameters["upper"] == 10
    assert weibull.with_parameters().cdf(9.0) == pytest.approx(weibull.cdf(9.0))
    assert beta.with_parameters().cdf(5.5) == pytest.approx(beta.cdf(5.5))
