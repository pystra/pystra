"""Failed inner FORM runs remain inspectable and stop dependent calculations."""

import pytest

import pystra as ra


def problem():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    model.add_variable(ra.Normal("Y", 0, 1))
    return model, ra.LimitState(lambda X, Y: 3 - X - Y)


@pytest.mark.parametrize("method", [ra.ImportanceSampling, ra.LineSampling])
def test_supplied_failed_form_keeps_its_record(method):
    model, limit_state = problem()
    form = ra.FORM(
        model,
        limit_state,
        options=ra.FORMOptions(max_iterations=1),
        on_failure="return",
    )
    with pytest.warns(RuntimeWarning, match="did not converge"):
        failed = form.run()
    with pytest.raises(ra.AnalysisError, match="successfully converged") as error:
        method(model, limit_state, form=form).run()
    assert error.value.result is failed
    assert failed.status == "not_converged"
    assert failed.n_limit_state_evaluations == 3
    assert failed.beta is None


@pytest.mark.parametrize("method", [ra.ImportanceSampling, ra.LineSampling])
@pytest.mark.parametrize(
    "state", ["never_run", "invalidated", "execution_failed", "changed"]
)
def test_supplied_form_does_not_fabricate_a_failure_record(method, state):
    model, limit_state = problem()
    form = ra.FORM(model, limit_state)
    if state != "never_run":
        form.run()
        if state == "execution_failed":

            def fail(X, Y):
                raise RuntimeError("solver stopped")

            limit_state.expression = fail
            with pytest.raises(ra.AnalysisError, match="solver stopped"):
                form.run()
        elif state == "changed":
            form.options = ra.FORMOptions(max_iterations=1)
            form.on_failure = "return"
            with pytest.warns(RuntimeWarning):
                form.run()
            model.variable("X").set_location(1)
        else:
            form._results_valid = False
    with pytest.raises(ra.AnalysisError, match="successfully converged") as error:
        method(model, limit_state, form=form).run()
    assert error.value.result is None


@pytest.mark.parametrize("method", [ra.ImportanceSampling, ra.LineSampling])
def test_failed_recomputation_keeps_new_record(method, monkeypatch):
    class ExtensionNormal(ra.Normal):
        pass

    model, limit_state = problem()
    model = ra.StochasticModel()
    model.add_variable(ExtensionNormal("X", 0, 1))
    model.add_variable(ra.Normal("Y", 0, 1))
    form = ra.FORM(model, limit_state, on_failure="return")
    first = form.run()
    assert first.converged and form._run_model_state is None
    run = form.run
    outcomes = []

    def recompute():
        form.options = ra.FORMOptions(max_iterations=1)
        outcomes.append(run())
        return outcomes[-1]

    monkeypatch.setattr(form, "run", recompute)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        with pytest.raises(ra.AnalysisError, match="successfully converged") as error:
            method(model, limit_state, form=form).run()
    assert len(outcomes) == 1
    assert error.value.result is outcomes[0]
    assert not error.value.result.converged
    assert first.converged


@pytest.mark.parametrize("method", ["numerical", "closed_form"])
@pytest.mark.parametrize("on_failure", ["return", "raise"])
def test_sensitivity_stops_at_failed_baseline(method, on_failure, monkeypatch):
    model, limit_state = problem()
    analysis = ra.SensitivityAnalysis(
        model,
        limit_state,
        method=method,
        on_failure=on_failure,
        options=ra.FORMOptions(max_iterations=1),
    )
    run = ra.FORM.run
    outcomes = []

    def record(form):
        outcomes.append(run(form))
        # No consumer may use an unconverged direction or index.
        form._alpha = form._beta = None
        return outcomes[-1]

    monkeypatch.setattr(ra.FORM, "run", record)
    with pytest.warns(RuntimeWarning):
        if on_failure == "return":
            result = analysis.run()
        else:
            with pytest.raises(ra.AnalysisError) as error:
                analysis.run()
            result = error.value.result
    assert len(outcomes) == 1
    assert result.form is outcomes[0]
    assert result.diagnostics["failed_form"] is outcomes[0]
    assert result.diagnostics["phase"] == "baseline"
    assert result.n_limit_state_evaluations == outcomes[0].n_limit_state_evaluations
    assert not result.converged and result.beta is None
    assert result.marginal == {} and result.correlation is None


def test_sensitivity_stops_at_failed_perturbation_and_resets_diagnostics(monkeypatch):
    model, limit_state = problem()
    analysis = ra.SensitivityAnalysis(model, limit_state, on_failure="return")
    run = ra.FORM.run
    outcomes = []

    def record(form):
        if form.model is not model:
            form.options = ra.FORMOptions(max_iterations=1)
        result = run(form)
        outcomes.append(result)
        if not result.converged:
            form._alpha = form._beta = None
        return result

    monkeypatch.setattr(ra.FORM, "run", record)
    with pytest.warns(RuntimeWarning):
        result = analysis.run()
    assert len(outcomes) == 2
    assert result.form is outcomes[0] and result.form.converged
    assert result.diagnostics["failed_form"] is outcomes[1]
    assert not outcomes[1].converged
    assert result.diagnostics["phase"] == "perturbation"
    assert result.diagnostics["variable"] == "X"
    assert result.diagnostics["parameter"] == "mean"
    assert result.diagnostics["step"] == pytest.approx(0.01)
    assert result.n_limit_state_evaluations == sum(
        r.n_limit_state_evaluations for r in outcomes
    )
    assert result.marginal == {} and result.beta is None
    with pytest.raises(TypeError):
        result.diagnostics["phase"] = "changed"
    monkeypatch.setattr(ra.FORM, "run", run)
    fresh = analysis.run()
    assert fresh.converged and fresh.diagnostics == {}
    assert result.diagnostics["phase"] == "perturbation"
