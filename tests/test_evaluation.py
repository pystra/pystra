"""Limit-state evaluation keeps no state and each analysis counts its own calls."""

import numpy as np
import pytest

import pystra as ra


def model():
    m = ra.StochasticModel()
    m.add_variable(ra.Normal("R", 10, 1))
    m.add_variable(ra.Normal("S", 5, 1))
    return m


def test_limit_state_keeps_no_evaluation_state():
    limit_state = ra.LimitState(lambda R, S: R - S)
    ra.FORM(model=model(), limit_state=limit_state).run()
    assert not any(hasattr(limit_state, name) for name in ("model", "options", "x"))


def test_finite_differences_leave_the_points_unchanged():
    limit_state = ra.LimitState(lambda R, S: R - S)
    x = np.array([[9.0], [4.0]])
    before = x.copy()
    G, gradient = limit_state.evaluate_lsf(x, model(), differentiation="ffd")
    assert np.array_equal(x, before)
    assert float(np.ravel(G)[0]) == pytest.approx(5.0)
    assert gradient[:, 0] == pytest.approx([1.0, -1.0])


def test_each_run_counts_its_own_evaluations():
    limit_state = ra.LimitState(lambda R, S: R - S)
    shared = model()
    first = ra.FORM(model=shared, limit_state=limit_state)
    first.run()
    second = ra.FORM(model=shared, limit_state=limit_state)
    second.run()
    assert first._n_evaluations == second._n_evaluations > 0
    assert shared.get_call_function() == 2 * first._n_evaluations
