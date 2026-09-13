"""Public reconstruction preserves the law, support, metadata and ownership."""

from types import MappingProxyType

import numpy as np
import pytest
from scipy import stats

import pystra as ra


def marginals():
    return [
        ra.Normal("X", 10, 2, start_point=8),
        ra.Lognormal("X", log_mean=1.7, log_std=0.4, start_point=8),
        ra.ShiftedLognormal("X", 10, 2, lower=3, start_point=8),
        ra.Gamma("X", rate=0.7, shape=4.3, start_point=8),
        ra.ChiSquare("X", df=7.3, start_point=8),
        ra.Gumbel("X", loc=3.7, scale=1.3, start_point=8),
        ra.GumbelMin("X", loc=13.7, scale=1.3, start_point=8),
        ra.Frechet("X", scale=10.1, shape=4.7, start_point=8),
        ra.Weibull("X", scale=3.7, shape=2.3, lower=2.1, start_point=8),
        ra.Beta("X", q=2.37, r=4.13, lower=3.7, upper=13.1, start_point=8),
        ra.Uniform("X", lower=3.7, upper=13.1, start_point=8),
        ra.ShiftedExponential("X", rate=0.7, shift=3.1, start_point=8),
        ra.ShiftedRayleigh("X", scale=3.7, shift=3.1, start_point=8),
        ra.GEV("X", shape=0.17, loc=3.7, scale=2.3, start_point=8),
        ra.GEVMin("X", shape=-0.17, loc=13.7, scale=2.3, start_point=8),
        ra.ScipyDist(
            "X", stats.truncnorm(-1.3, 2.7, loc=3.1, scale=2.3), start_point=8
        ),
        ra.Maximum("X", ra.Frechet("parent", scale=3.7, shape=4.3), 7.3, start_point=8),
        ra.MaxParent(
            "X", ra.Uniform("maximum", lower=2.1, upper=8.3), 7.3, start_point=8
        ),
        ra.ZeroInflated("X", ra.Lognormal("parent", 10, 2), 0.3, start_point=8),
    ]


@pytest.mark.parametrize("dist", marginals(), ids=lambda d: type(d).__name__)
def test_reconstruction_roundtrip(dist):
    probabilities = np.array([1e-8, 0.01, 0.3, 0.7, 0.99, 1 - 1e-8])
    x = dist.ppf(probabilities)
    for clone in (type(dist)(**dist.parameters), dist.with_parameters()):
        assert clone is not dist
        assert type(clone) is type(dist)
        assert clone.name == dist.name
        assert clone.start_point == dist.start_point
        np.testing.assert_allclose(
            [clone.mean, clone.std], [dist.mean, dist.std], rtol=2e-14
        )
        for method in ("pdf", "cdf", "sf", "logcdf", "logsf"):
            np.testing.assert_allclose(
                getattr(clone, method)(x),
                getattr(dist, method)(x),
                rtol=2e-13,
                atol=2e-14,
            )
        np.testing.assert_allclose(clone.ppf(probabilities), x, rtol=2e-14)
        np.testing.assert_allclose(
            clone.ppf([0.0, 1.0]), dist.ppf([0.0, 1.0]), rtol=2e-14
        )
    with pytest.raises(TypeError):
        dist.parameters["name"] = "changed"
    with pytest.raises(TypeError, match="Unknown"):
        dist.with_parameters(typo=3)


@pytest.mark.parametrize(
    "dist",
    [
        ra.Beta("X", q=2.37, r=4.13, lower=3, upper=10),
        ra.Weibull("X", scale=3.7, shape=2.3, lower=2),
        ra.GEVMin("X", shape=0.17, loc=13.7, scale=2.3),
    ],
)
def test_native_copies_preserve_exact_frozen_parameters(dist):
    clone = dist.with_parameters()
    assert clone.dist_obj.args == dist.dist_obj.args
    assert clone.dist_obj.kwds == dist.dist_obj.kwds


def test_moment_replacement_holds_bounds_and_start_point():
    dist = ra.Weibull("X", 10, 2, lower=3, start_point=8)
    clone = dist.with_parameters(mean=11)
    assert clone.mean == pytest.approx(11)
    assert clone.std == pytest.approx(dist.std)
    assert clone.parameters["lower"] == 3
    assert clone.start_point == 8
    assert dist.mean == pytest.approx(10)
    assert clone.with_parameters(start_point=None).start_point == clone.mean
    with pytest.raises(TypeError, match="moments or native"):
        dist.with_parameters(mean=12, scale=3)


def test_replacing_shape_in_sensitivity_coordinates_holds_moments():
    dist = ra.GEV("X", 10, 2, shape=0.1)
    clone = dist.with_parameters(**{**dist.sensitivity_params, "shape": 0.2})
    np.testing.assert_allclose(
        [clone.mean, clone.std], [dist.mean, dist.std], rtol=2e-14
    )
    assert clone.shape == 0.2


@pytest.mark.parametrize(
    "dist, nested",
    [
        (ra.Maximum("X", ra.Normal("P", 10, 2), 2), "parent"),
        (ra.MaxParent("X", ra.Normal("P", 10, 2), 2), "max_dist"),
        (ra.ZeroInflated("X", ra.Normal("P", 10, 2), 0.3), "dist"),
    ],
)
def test_composite_parameter_snapshot_and_copy_are_independent(dist, nested):
    parameters = dist.parameters
    parameters[nested].set_location(30)
    clone = dist.with_parameters()
    getattr(clone, nested).set_location(40)
    assert getattr(dist, nested).mean == 10
    assert dist.sensitivity_params == {}
    with pytest.raises(TypeError, match="Unknown"):
        dist.with_parameters(mean=20)


def test_scipy_snapshot_and_replacement_are_independent():
    dist = ra.ScipyDist("X", stats.norm(loc=10, scale=2))
    parameters = dist.parameters
    parameters["dist_obj"].kwds["loc"] = 30
    clone = dist.with_parameters(dist_obj=stats.norm(loc=20, scale=3))
    clone.set_location(40)
    assert dist.mean == 10
    assert dist.parameters["dist_obj"].mean() == 10
    assert clone.mean == 40


def test_custom_distribution_declares_public_parameters():
    class BoundedNormal(ra.Distribution):
        def __init__(self, name, mean, std, *, lower, start_point=None):
            self.lower = lower
            super().__init__(name, mean=mean, std=std, start_point=start_point)
            self.dist_obj = stats.truncnorm(
                (lower - mean) / std, np.inf, loc=mean, scale=std
            )

        @property
        def parameters(self):
            return MappingProxyType(
                {
                    "name": self.name,
                    "mean": self.mean,
                    "std": self.std,
                    "lower": self.lower,
                    "start_point": self.start_point,
                }
            )

    dist = BoundedNormal("X", 10, 2, lower=3, start_point=9)
    clone = dist.with_parameters(mean=11)
    assert clone.lower == 3
    assert clone.mean == 11
    assert clone.start_point == 9
    assert dist.mean == 10
