"""PySTRA's exception hierarchy."""

import pytest

import pystra as ra


def test_errors_form_a_hierarchy_compatible_with_builtins():
    assert issubclass(ra.ModelError, ra.PystraError)
    assert issubclass(ra.ModelError, ValueError)
    assert issubclass(ra.AnalysisError, ra.PystraError)
    assert issubclass(ra.AnalysisError, RuntimeError)


def test_invalid_model_input_raises_model_error():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    with pytest.raises(ra.ModelError, match="already exists"):
        model.add_variable(ra.Normal("X", 0, 1))
    with pytest.raises(ra.ModelError, match="Distribution or Constant"):
        model.add_variable(object())


@pytest.mark.parametrize(
    "build",
    [
        lambda: ra.Normal("N", 0, -1),
        lambda: ra.Normal("N", 0, 0),
        lambda: ra.ZeroInflated("Z", ra.Normal("N", 1, 1), -0.1),
        lambda: ra.FBCProcess("Q", ra.Normal("Q", 1, 1), basic_interval=-1),
    ],
    ids=["negative std", "zero std", "negative zero probability", "negative interval"],
)
def test_invalid_distribution_input_is_a_value_error(build):
    with pytest.raises(ValueError) as info:
        build()
    assert isinstance(info.value, ra.ModelError)


def test_analysis_error_keeps_the_failed_result():
    error = ra.AnalysisError("did not converge", result="failed result")
    assert str(error) == "did not converge"
    assert error.result == "failed result"
