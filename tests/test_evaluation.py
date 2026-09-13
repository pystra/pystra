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
    G, gradient = limit_state._evaluate_lsf(x, model(), differentiation="ffd")
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


@pytest.mark.parametrize("mode", ["no", "ffd", "ddm"])
@pytest.mark.parametrize(
    "points",
    [
        [9.0, 4.0],
        [[9.0, 4.0]],
        [[9.0, 4.0], [3.0, 8.0]],
        [[9.0, 4.0], [3.0, 8.0], [2.0, 2.0]],
    ],
)
def test_public_evaluator_point_and_row_batches(mode, points):
    # Unequal derivatives and a square batch detect accidental transposition.
    problem = model()
    problem.add_variable(ra.Constant("factor", 2.0))
    expression = lambda S, factor, R: (
        factor * R - 3 * S,
        [factor, -3 * np.ones_like(S)],
    )
    limit_state = ra.LimitState(expression)
    x = np.asarray(points)
    before = x.copy()
    values, gradients = limit_state.evaluate(
        x, problem, differentiation=mode, block_size=2
    )
    np.testing.assert_array_equal(x, before)
    np.testing.assert_allclose(values, 2 * x[..., 0] - 3 * x[..., 1])
    expected = [0, 0] if mode == "no" else [2, -3]
    np.testing.assert_allclose(
        gradients, np.broadcast_to(expected, x.shape), atol=1e-10
    )
    assert np.shape(values) == x.shape[:-1]
    assert gradients.shape == x.shape
    assert isinstance(values, float) if x.ndim == 1 else isinstance(values, np.ndarray)


@pytest.mark.parametrize("points", [1.0, [1.0], [[1.0], [2.0]], np.zeros((1, 2, 1))])
def test_public_evaluator_rejects_wrong_shapes(points):
    with pytest.raises(ra.ModelError, match="shape"):
        ra.LimitState(lambda R, S: R - S).evaluate(points, model())


@pytest.mark.parametrize("mode", ["no", "ffd", "ddm"])
def test_public_evaluator_empty_batch(mode):
    values, gradients = ra.LimitState(lambda R, S: R - S).evaluate(
        np.empty((0, 2)), model(), differentiation=mode
    )
    assert values.shape == (0,)
    assert gradients.shape == (0, 2)


@pytest.mark.parametrize(
    "settings",
    [
        {"block_size": 0},
        {"block_size": 1.5},
        {"ffd_parameter": 0},
        {"ffd_parameter": np.inf},
    ],
)
def test_evaluator_rejects_invalid_settings(settings):
    with pytest.raises(ra.ModelError):
        ra.LimitState(lambda R, S: R - S).evaluate([1, 2], model(), **settings)
