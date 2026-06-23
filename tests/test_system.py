"""Tests for system reliability limit-state composition."""

import numpy as np
import pytest

import pystra as ra


def test_component_filters_unused_model_variables():
    component = ra.Component("bending", lambda R, M: R - M)

    result = component.evaluate(
        R=np.array([10.0, 4.0]),
        M=np.array([6.0, 5.0]),
        unused=np.array([100.0, 100.0]),
    )

    np.testing.assert_allclose(result, np.array([4.0, -1.0]))


def test_component_accepts_limit_state_instance():
    limit_state = ra.LimitState(lambda R, S: R - S)
    component = ra.Component("member", limit_state)

    assert component.as_limit_state() is limit_state
    np.testing.assert_allclose(
        component.evaluate(R=np.array([5.0]), S=np.array([3.0])),
        np.array([2.0]),
    )


def test_series_system_uses_minimum_child_limit_state():
    system = ra.SeriesSystem(
        [
            ra.Component("flexure", lambda R, S: R - S),
            ra.Component("shear", lambda V, S: V - 2.0 * S),
        ]
    )

    result = system.evaluate(
        R=np.array([10.0, 10.0]),
        V=np.array([14.0, 6.0]),
        S=np.array([5.0, 4.0]),
    )

    np.testing.assert_allclose(result, np.array([4.0, -2.0]))
    np.testing.assert_array_equal(
        system.failure_mask(
            R=np.array([10.0, 10.0]),
            V=np.array([14.0, 6.0]),
            S=np.array([5.0, 4.0]),
        ),
        [False, True],
    )


def test_parallel_system_uses_maximum_child_limit_state():
    system = ra.ParallelSystem(
        [
            ra.Component("support_a", lambda A: A - 1.0),
            ra.Component("support_b", lambda B: B - 1.0),
        ]
    )

    result = system.evaluate(A=np.array([2.0, 0.0]), B=np.array([3.0, 0.5]))

    np.testing.assert_allclose(result, np.array([2.0, -0.5]))
    np.testing.assert_array_equal(
        system.failure_mask(A=np.array([2.0, 0.0]), B=np.array([3.0, 0.5])),
        [False, True],
    )


def test_k_of_n_system_fails_when_k_children_fail():
    system = ra.KofNSystem(
        [
            ra.Component("a", lambda A: A),
            ra.Component("b", lambda B: B),
            ra.Component("c", lambda C: C),
        ],
        k=2,
    )

    result = system.evaluate(
        A=np.array([1.0, -1.0, -1.0, -1.0]),
        B=np.array([1.0, 1.0, -1.0, -1.0]),
        C=np.array([1.0, 1.0, 1.0, -1.0]),
    )

    np.testing.assert_allclose(result, np.array([1.5, 0.5, -0.5, -1.5]))
    np.testing.assert_array_equal(
        result < 0.0,
        np.array([False, False, True, True]),
    )


def test_k_of_n_system_rejects_invalid_k():
    with pytest.raises(ValueError, match="1 <= k"):
        ra.KofNSystem([ra.Component("a", lambda A: A)], k=0)


def test_nested_system_composition():
    system = ra.SeriesSystem(
        [
            ra.Component("main", lambda G: G),
            ra.ParallelSystem(
                [
                    ra.Component("backup_a", lambda A: A),
                    ra.Component("backup_b", lambda B: B),
                ]
            ),
        ]
    )

    result = system.evaluate(
        G=np.array([2.0, 2.0, -1.0]),
        A=np.array([1.0, -2.0, 1.0]),
        B=np.array([-1.0, -3.0, 1.0]),
    )

    # Equivalent system LSF is min(G, max(A, B)).
    np.testing.assert_allclose(result, np.array([1.0, -2.0, -1.0]))
    assert [component.name for component in system.components] == [
        "main",
        "backup_a",
        "backup_b",
    ]


def test_cut_set_system_matches_boolean_cut_sets():
    components = {
        name: ra.Component(name, lambda var=name, **kwargs: kwargs[var])
        for name in ("E1", "E2", "E3", "E4", "E5")
    }
    system = ra.CutSetSystem(
        [["E1", "E2"], ["E3", "E4"], ["E3", "E5"]],
        components=components,
    )

    failed = {
        "E1": np.array([False, True, True, False]),
        "E2": np.array([False, True, False, True]),
        "E3": np.array([True, False, True, True]),
        "E4": np.array([False, False, True, False]),
        "E5": np.array([False, False, False, True]),
    }
    values = {name: np.where(mask, -1.0, 1.0) for name, mask in failed.items()}

    expected = (
        (failed["E1"] & failed["E2"])
        | (failed["E3"] & failed["E4"])
        | (failed["E3"] & failed["E5"])
    )

    np.testing.assert_array_equal(system.failure_mask(**values), expected)
    assert set(system.component_values(**values)) == set(components)


def test_tie_set_system_matches_boolean_tie_sets():
    components = {
        name: ra.Component(name, lambda var=name, **kwargs: kwargs[var])
        for name in ("A", "B", "C", "D")
    }
    system = ra.TieSetSystem([["A", "B"], ["C", "D"]], components=components)

    safe = {
        "A": np.array([True, True, False, False]),
        "B": np.array([True, False, True, False]),
        "C": np.array([False, True, True, False]),
        "D": np.array([False, True, False, True]),
    }
    values = {name: np.where(mask, 1.0, -1.0) for name, mask in safe.items()}

    expected_failure = ~((safe["A"] & safe["B"]) | (safe["C"] & safe["D"]))

    np.testing.assert_array_equal(system.failure_mask(**values), expected_failure)


def test_system_as_limit_state_integrates_with_pystra_evaluation():
    system = ra.SeriesSystem(
        [
            ra.Component("flexure", lambda R, S: R - S),
            ra.Component("shear", lambda V, S: V - 2.0 * S),
        ]
    )
    limit_state = system.as_limit_state()

    model = ra.StochasticModel()
    model.addVariable(ra.Normal("R", 10.0, 1.0))
    model.addVariable(ra.Normal("V", 12.0, 1.0))
    model.addVariable(ra.Normal("S", 4.0, 1.0))

    options = ra.AnalysisOptions()
    options.setPrintOutput(False)

    x = np.array(
        [
            [10.0, 10.0],
            [12.0, 6.0],
            [4.0, 4.0],
        ]
    )
    values, gradient = limit_state.evaluate_lsf(x, model, options, diff_mode="no")

    np.testing.assert_allclose(values, np.array([[4.0, -2.0]]))
    assert gradient.shape == x.shape


def test_system_limit_state_runs_form_analysis():
    system = ra.SeriesSystem(
        [
            ra.Component("flexure", lambda R, S: R - S),
            ra.Component("shear", lambda V, S: V - 2.0 * S),
        ]
    )

    model = ra.StochasticModel()
    model.addVariable(ra.Normal("R", 10.0, 1.0))
    model.addVariable(ra.Normal("V", 12.0, 1.0))
    model.addVariable(ra.Normal("S", 4.0, 1.0))

    options = ra.AnalysisOptions()
    options.setPrintOutput(False)

    form = ra.Form(
        stochastic_model=model,
        limit_state=system.as_limit_state(),
        analysis_options=options,
    )
    form.run()

    assert np.isfinite(form.getBeta())
    assert np.all(form.getFailure() > 0.0)


def test_component_values_and_failure_masks():
    system = ra.SeriesSystem(
        [
            ra.Component("a", lambda A: A),
            ra.Component("b", lambda B: B),
        ]
    )

    values = system.component_values(
        A=np.array([1.0, -1.0]),
        B=np.array([-2.0, 3.0]),
    )
    masks = system.component_failure_masks(
        A=np.array([1.0, -1.0]),
        B=np.array([-2.0, 3.0]),
    )

    np.testing.assert_allclose(values["a"], np.array([1.0, -1.0]))
    np.testing.assert_allclose(values["b"], np.array([-2.0, 3.0]))
    np.testing.assert_array_equal(masks["a"], np.array([False, True]))
    np.testing.assert_array_equal(masks["b"], np.array([True, False]))


def test_duplicate_component_names_are_rejected_for_component_value_mapping():
    system = ra.SeriesSystem(
        [
            ra.Component("member", lambda A: A),
            ra.Component("member", lambda B: B),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate component name"):
        system.component_values(A=np.array([1.0]), B=np.array([2.0]))


def test_ditlevsen_bounds_are_exact_for_two_events():
    lower, upper = ra.ditlevsen_bounds(
        [0.1, 0.2],
        {(0, 1): 0.03},
    )

    assert lower == pytest.approx(0.27)
    assert upper == pytest.approx(0.27)


def test_ditlevsen_upper_bound_matches_maincon_identical_series_benchmark():
    n_events = 100
    component_probability = 1.969e-3
    pair_probability = 1.361e-3
    probabilities = np.full(n_events, component_probability)
    intersections = np.full((n_events, n_events), pair_probability)
    np.fill_diagonal(intersections, component_probability)

    lower, upper = ra.ditlevsen_bounds(probabilities, intersections)

    assert lower == pytest.approx(2.577e-3)
    assert upper == pytest.approx(6.2161e-2)


def test_ditlevsen_bounds_can_optimize_event_ordering():
    probabilities = np.array([0.08, 0.07, 0.06])
    intersections = np.array(
        [
            [0.08, 0.03, 0.01],
            [0.03, 0.07, 0.02],
            [0.01, 0.02, 0.06],
        ]
    )

    lower, upper = ra.ditlevsen_bounds(
        probabilities,
        intersections,
        optimize_order=True,
    )

    assert lower == pytest.approx(0.15)
    assert upper == pytest.approx(0.16)


def test_empty_system_raises():
    with pytest.raises(ValueError, match="at least one child"):
        ra.SeriesSystem([])
