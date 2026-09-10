"""Distribution, constant and model accessors after the 2.0 renames."""

import pytest

import pystra as ra


def model():
    result = ra.StochasticModel()
    result.add_variable(ra.Normal("R", 10.0, std=1.0, start_point=9.0))
    result.add_variable(ra.Normal("S", 5.0, 1.0))
    result.add_variable(ra.Constant("c", value=2.0))
    return result


def test_distribution_moments_are_read_only_properties():
    R = ra.Normal("R", 10.0, std=1.0)
    assert (R.mean, R.std, R.start_point) == (10.0, 1.0, 10.0)
    with pytest.raises(AttributeError):
        R.mean = 11.0
    with pytest.raises(AttributeError):
        R.std = 2.0
    assert ra.Normal("R", 10.0, 1.0, start_point=9.0).start_point == 9.0


def test_old_accessors_are_removed():
    R = ra.Normal("R", 10.0, 1.0)
    for name in ("get_mean", "get_stdv", "get_start_point", "stdv", "startpoint"):
        assert not hasattr(R, name)
    assert not hasattr(ra.Constant("c", 1.0), "get_value")
    assert not hasattr(ra.StochasticModel(), "get_constants")
    assert not hasattr(ra.StochasticModel(), "get_variable")


def test_constant_value_and_model_accessors():
    m = model()
    assert ra.Constant("c", value=2.0).value == 2.0
    assert m.variable("R").mean == 10.0 and m.variable("R").start_point == 9.0
    assert dict(m.constants) == {"c": 2.0}
    with pytest.raises(TypeError):
        m.constants["c"] = 3.0


def test_reprs_describe_the_objects():
    m = model()
    assert repr(m.variable("R")) == "Normal('R', mean=10, std=1)"
    assert repr(ra.Constant("c", value=2.0)) == "Constant('c', value=2.0)"
    assert repr(m) == (
        "StochasticModel([Normal('R', mean=10, std=1), Normal('S', mean=5, std=1), "
        "Constant('c', value=2.0)])"
    )
    form = ra.FORM(m, ra.LimitState(lambda R, S, c: c * R - S))
    assert repr(form).startswith("FORM(variables=('R', 'S'), options=FORMOptions(")
    sensitivity = ra.SensitivityAnalysis(m, ra.LimitState(lambda R, S, c: c * R - S))
    assert "method='numerical'" in repr(sensitivity)
