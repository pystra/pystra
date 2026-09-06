"""SORM must use a converged FORM design point for either fitting method."""

import numpy as np
import pytest
from scipy.stats import norm

import pystra as ra


@pytest.fixture
def linear_problem():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    model.add_variable(ra.Normal("Y", 0, 1))
    return model, ra.LimitState(lambda X, Y: 5 - X - Y), ra.AnalysisOptions()


def assert_invalid(analysis):
    assert not analysis.results_valid
    assert analysis.betaHL is None
    assert analysis.kappa is None
    assert analysis.kappa_pf is None
    assert analysis.pf2_breitung is None
    assert analysis.betag_breitung is None
    assert analysis.pf2_breitung_m is None
    assert analysis.betag_breitung_m is None
    for report in (analysis.show_results, analysis.show_detailed_output):
        with pytest.raises(ValueError, match="Analysis not yet run"):
            report()


@pytest.mark.parametrize("fit_type", ["cf", "pf"])
def test_sorm_rejects_automatically_run_nonconverged_form(linear_problem, fit_type):
    model, limit_state, options = linear_problem
    options.set_imax(1)
    with pytest.warns(RuntimeWarning, match="FORM did not converge"):
        analysis = ra.Sorm(model, limit_state, options)

    assert not analysis.form.converged
    calls = model.get_call_function()
    with pytest.raises(RuntimeError, match="successfully converged FORM"):
        analysis.run(fit_type)

    assert model.get_call_function() == calls
    assert_invalid(analysis)


@pytest.mark.parametrize("fit_type", ["cf", "pf"])
@pytest.mark.parametrize("run_form", [False, True])
def test_sorm_rejects_supplied_invalid_form(linear_problem, fit_type, run_form):
    model, limit_state, options = linear_problem
    options.set_imax(1)
    form = ra.Form(model, limit_state, options)
    if run_form:
        with pytest.warns(RuntimeWarning, match="FORM did not converge"):
            form.run()
    analysis = ra.Sorm(model, limit_state, options, form=form)

    with pytest.raises(RuntimeError, match="successfully converged FORM"):
        analysis.run(fit_type)
    assert_invalid(analysis)


@pytest.mark.parametrize("fit_type", ["cf", "pf"])
def test_sorm_failed_form_rerun_clears_results_and_can_recover(
    linear_problem, fit_type
):
    model, limit_state, options = linear_problem
    analysis = ra.Sorm(model, limit_state, options)
    analysis.run(fit_type)
    expected_beta = 5 / np.sqrt(2)
    expected_pf = norm.sf(expected_beta)
    assert analysis.results_valid
    assert analysis.pf2_breitung == pytest.approx(expected_pf, rel=1e-6)

    options.set_imax(1)
    with pytest.warns(RuntimeWarning, match="FORM did not converge"):
        analysis.form.run()
    with pytest.raises(RuntimeError, match="successfully converged FORM"):
        analysis.run(fit_type)
    assert_invalid(analysis)

    options.set_imax(100)
    analysis.form.run()
    analysis.run(fit_type)
    assert analysis.results_valid
    assert analysis.betag_breitung == pytest.approx(expected_beta, rel=1e-6)
    assert analysis.pf2_breitung == pytest.approx(expected_pf, rel=1e-6)
    assert analysis.pf2_breitung_m == pytest.approx(expected_pf, rel=1e-6)


@pytest.mark.parametrize("method", ["run_curvefit", "run_pointfit"])
def test_direct_fitting_methods_check_form_and_preserve_reporting(
    linear_problem, method, capsys
):
    model, limit_state, options = linear_problem
    analysis = ra.Sorm(model, limit_state, options)
    options.set_print_output(True)
    getattr(analysis, method)()
    assert analysis.results_valid
    assert "SECOND ORDER RELIABILITY METHOD" in capsys.readouterr().out

    options.set_print_output(False)
    options.set_imax(1)
    with pytest.warns(RuntimeWarning, match="FORM did not converge"):
        analysis.form.run()
    with pytest.raises(RuntimeError, match="successfully converged FORM"):
        getattr(analysis, method)()
    assert_invalid(analysis)


def test_invalid_fit_type_invalidates_previous_results(linear_problem):
    model, limit_state, options = linear_problem
    analysis = ra.Sorm(model, limit_state, options)
    analysis.run()
    with pytest.raises(ValueError, match="Unknown fit_type"):
        analysis.run("invalid")
    assert_invalid(analysis)


@pytest.mark.parametrize("fit_type", ["cf", "pf"])
def test_fitting_exception_does_not_leave_valid_results(linear_problem, fit_type):
    model, limit_state, options = linear_problem
    analysis = ra.Sorm(model, limit_state, options)
    analysis.run()

    def failed_evaluation(X, Y):
        raise RuntimeError("External solver failed")

    limit_state.set_expression(failed_evaluation)
    with pytest.raises(RuntimeError, match="External solver failed"):
        analysis.run(fit_type)
    assert_invalid(analysis)
