"""SORM must use a converged FORM design point for either fitting method."""

import numpy as np
import pytest
from scipy.stats import norm

import pystra as ra

FITS = ["curve", "point"]


@pytest.fixture
def linear_problem():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("X", 0, 1))
    model.add_variable(ra.Normal("Y", 0, 1))
    return model, ra.LimitState(lambda X, Y: 5 - X - Y)


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


@pytest.mark.parametrize("fit", FITS)
def test_sorm_rejects_automatically_run_nonconverged_form(linear_problem, fit):
    model, limit_state = linear_problem
    options = ra.SORMOptions(fit=fit, form=ra.FORMOptions(max_iterations=1))
    analysis = ra.SORM(model, limit_state, options=options)
    with pytest.warns(RuntimeWarning, match="FORM did not converge"):
        with pytest.raises(RuntimeError, match="successfully converged FORM"):
            analysis.run()

    assert not analysis.form.converged
    # Only FORM evaluated the limit state; SORM stopped before fitting.
    assert model.get_call_function() == analysis.form.get_no_function_calls()
    assert_invalid(analysis)


@pytest.mark.parametrize("fit", FITS)
@pytest.mark.parametrize("run_form", [False, True])
def test_sorm_rejects_supplied_invalid_form(linear_problem, fit, run_form):
    model, limit_state = linear_problem
    form = ra.FORM(model, limit_state, options=ra.FORMOptions(max_iterations=1))
    if run_form:
        with pytest.warns(RuntimeWarning, match="FORM did not converge"):
            form.run()
    analysis = ra.SORM(model, limit_state, options=ra.SORMOptions(fit=fit), form=form)

    with pytest.raises(RuntimeError, match="successfully converged FORM"):
        analysis.run()
    assert_invalid(analysis)


def test_supplied_form_keeps_its_own_settings(linear_problem):
    model, limit_state = linear_problem
    form = ra.FORM(model, limit_state)
    options = ra.SORMOptions(form=ra.FORMOptions(max_iterations=5))
    with pytest.raises(ValueError, match="supplied FORM analysis keeps its own"):
        ra.SORM(model, limit_state, options=options, form=form)


@pytest.mark.parametrize("fit", FITS)
def test_sorm_failed_form_rerun_clears_results_and_can_recover(linear_problem, fit):
    model, limit_state = linear_problem
    form = ra.FORM(model, limit_state)
    form.run()
    analysis = ra.SORM(model, limit_state, options=ra.SORMOptions(fit=fit), form=form)
    analysis.run()
    expected_beta = 5 / np.sqrt(2)
    expected_pf = norm.sf(expected_beta)
    assert analysis.results_valid
    assert analysis.pf2_breitung == pytest.approx(expected_pf, rel=1e-6)

    form.options = ra.FORMOptions(max_iterations=1)
    with pytest.warns(RuntimeWarning, match="FORM did not converge"):
        form.run()
    with pytest.raises(RuntimeError, match="successfully converged FORM"):
        analysis.run()
    assert_invalid(analysis)

    form.options = ra.FORMOptions()
    form.run()
    analysis.run()
    assert analysis.results_valid
    assert analysis.betag_breitung == pytest.approx(expected_beta, rel=1e-6)
    assert analysis.pf2_breitung == pytest.approx(expected_pf, rel=1e-6)
    assert analysis.pf2_breitung_m == pytest.approx(expected_pf, rel=1e-6)


@pytest.mark.parametrize("method", ["run_curvefit", "run_pointfit"])
def test_direct_fitting_methods_check_form_and_preserve_reporting(
    linear_problem, method, capsys
):
    model, limit_state = linear_problem
    analysis = ra.SORM(model, limit_state)
    getattr(analysis, method)()
    assert analysis.results_valid
    analysis.show_results()
    assert "SECOND ORDER RELIABILITY METHOD" in capsys.readouterr().out

    analysis.form.options = ra.FORMOptions(max_iterations=1)
    with pytest.warns(RuntimeWarning, match="FORM did not converge"):
        analysis.form.run()
    with pytest.raises(RuntimeError, match="successfully converged FORM"):
        getattr(analysis, method)()
    assert_invalid(analysis)


def test_invalid_fit_is_rejected_by_the_options():
    with pytest.raises(ValueError, match="fit must be one of"):
        ra.SORMOptions(fit="invalid")


@pytest.mark.parametrize("fit", FITS)
def test_fitting_exception_does_not_leave_valid_results(linear_problem, fit):
    model, limit_state = linear_problem
    analysis = ra.SORM(model, limit_state, options=ra.SORMOptions(fit=fit))
    analysis.run()

    def failed_evaluation(X, Y):
        raise RuntimeError("External solver failed")

    limit_state.set_expression(failed_evaluation)
    with pytest.raises(RuntimeError, match="External solver failed"):
        analysis.run()
    assert_invalid(analysis)
