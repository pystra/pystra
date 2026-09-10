"""Nonconvergence raises AnalysisError unless on_failure="return"."""

import pytest

import pystra as ra


def normal_model():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("R", 10.0, 1.0))
    model.add_variable(ra.Normal("S", 5.0, 1.0))
    return model


def margin():
    return ra.LimitState(lambda R, S: R - S)


def stopped():
    return ra.FORMOptions(max_iterations=1)


def test_form_raises_with_the_unconverged_record():
    analysis = ra.FORM(normal_model(), margin(), options=stopped())
    with pytest.raises(ra.AnalysisError, match="did not converge") as info:
        analysis.run()
    assert isinstance(info.value, RuntimeError)
    assert isinstance(info.value, ra.PystraError)
    record = info.value.result
    assert isinstance(record, ra.FORMResult) and record.status == "not_converged"
    assert record.failure_probability is None and record.iterations == 1


def test_form_returns_the_unconverged_record_on_request():
    analysis = ra.FORM(normal_model(), margin(), options=stopped(), on_failure="return")
    with pytest.warns(RuntimeWarning, match="did not converge"):
        record = analysis.run()
    assert record.status == "not_converged" and record.beta is None


def test_unknown_failure_policy_is_rejected():
    with pytest.raises(ValueError, match="on_failure"):
        ra.FORM(normal_model(), margin(), on_failure="ignore")


def curved(c, **kwargs):
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X1", 0.0, 1.0))
    model.add_variable(ra.Normal("X2", 0.0, 1.0))
    limit_state = ra.LimitState(lambda X1, X2: 3 - X2 - c * X1**2)
    form = ra.FORM(model, limit_state)
    form.run()
    return ra.SORM(model, limit_state, form=form, **kwargs)


def test_sorm_raises_when_breitung_is_undefined():
    with pytest.raises(ra.AnalysisError, match="undefined") as info:
        curved(0.2).run()
    assert info.value.result.status == "not_converged"
    assert info.value.result.form.converged


def test_sorm_failure_returns_a_record_without_an_estimate():
    options = ra.SORMOptions(form=stopped())
    analysis = ra.SORM(normal_model(), margin(), options=options, on_failure="return")
    with pytest.warns(RuntimeWarning, match="did not converge"):
        record = analysis.run()
    assert record.status == "not_converged" and "converged FORM" in record.message
    assert record.failure_probability is None and record.curvatures is None
    assert not record.form.converged
    assert "unavailable" in record.summary()


def test_system_form_failure_keeps_the_component_records():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0.0, 1.0))
    model.add_variable(ra.Normal("Y", 0.0, 1.0))
    system = ra.SeriesSystem(
        [ra.Component("x", lambda X: 3 - X), ra.Component("y", lambda Y: 3 - Y)]
    )
    analysis = ra.SystemFORM(model, system, options=stopped(), on_failure="return")
    with pytest.warns(RuntimeWarning, match="did not converge"):
        record = analysis.run()
    assert record.status == "not_converged" and record.failure_probability is None
    assert list(record.component_results) == ["x"]
    assert not record.component_results["x"].converged
    assert "unavailable" in record.summary()
    with pytest.warns(RuntimeWarning), pytest.raises(ra.AnalysisError) as info:
        ra.SystemFORM(model, system, options=stopped()).run()
    assert info.value.result.status == "not_converged"


def test_sensitivity_failure_withholds_the_derivatives():
    analysis = ra.SensitivityAnalysis(
        normal_model(), margin(), options=stopped(), on_failure="return"
    )
    with pytest.warns(RuntimeWarning, match="did not converge"):
        record = analysis.run()
    assert record.status == "not_converged" and record.beta is None
    assert dict(record.marginal) == {} and record.correlation is None
    with pytest.warns(RuntimeWarning), pytest.raises(ra.AnalysisError):
        ra.SensitivityAnalysis(normal_model(), margin(), options=stopped()).run()
