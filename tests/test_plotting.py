"""Check geometry, coordinates and diagnostic semantics of public figures."""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pystra as ra
from pystra.active_learning import (
    ActiveLearningResult,
    LearningStep,
    PCESurrogate,
    ReliabilityEstimate,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def normal_form():
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("R", 0, 1))
    model.add_variable(ra.Normal("S", 0, 1))
    analysis = ra.FORM(
        model=model,
        limit_state=ra.LimitState(lambda R, S: 3 - R - 2 * S),
    )
    return analysis, analysis.run()


def test_contour_uses_rowwise_coordinates_and_bounded_batches():
    batches = []

    def response(points):
        batches.append(len(points))
        return 1 - points[:, 0] - 2 * points[:, 1]

    fig, existing = plt.subplots()
    returned, ax = ra.plotting.plot_limit_state(
        response,
        bounds=((-2, 3), (-4, 5)),
        n_points=17,
        batch_size=23,
        shade_failure=False,
        ax=existing,
    )
    assert returned is fig and ax is existing
    assert max(batches) <= 23 and sum(batches) == 17**2
    # get_paths() has the same meaning in every supported Matplotlib
    boundary = np.concatenate([path.vertices for path in ax.collections[0].get_paths()])
    np.testing.assert_allclose(1 - boundary[:, 0] - 2 * boundary[:, 1], 0, atol=1e-14)
    np.testing.assert_allclose(ax.get_xlim(), [-2, 3])
    np.testing.assert_allclose(ax.get_ylim(), [-4, 5])


def test_form_tangent_is_orthogonal_and_plotting_does_not_evaluate():
    analysis, result = normal_form()
    count = analysis.model.get_call_function()
    _, ax = ra.plotting.plot_form_geometry(result)
    tangent = ax.lines[0].get_xydata()
    alpha = np.asarray(result.alpha)
    np.testing.assert_allclose(
        tangent @ alpha, np.dot(result.design_point_u, alpha), atol=1e-14
    )
    assert analysis.model.get_call_function() == count
    with pytest.raises(ValueError, match="converged two-dimensional"):
        ra.plotting.plot_form_geometry(replace(result, status="not_converged"))
    with pytest.raises(ValueError, match="converged two-dimensional"):
        ra.plotting.plot_form_geometry(replace(result, design_point_u=(1, 2, 3)))


def fitted_pce():
    points = np.random.default_rng(73).normal(size=(40, 2))
    surrogate = PCESurrogate(degree=2, seed=42)
    surrogate.fit(points, 2 - points[:, 0] ** 2 + points[:, 1])
    return surrogate


def test_surrogate_slice_preserves_model_and_input():
    surrogate = fitted_pce()
    points = np.column_stack((np.linspace(-3, 3, 40), np.full(40, 0.7)))
    original = points.copy()
    fit = surrogate.fit_result
    mean, std = surrogate.predict(points)
    _, ax = ra.plotting.plot_surrogate_slice(
        surrogate,
        points,
        limit_state=lambda points: 2 - points[:, 0] ** 2 + points[:, 1],
        batch_size=7,
    )
    np.testing.assert_allclose(ax.lines[1].get_ydata(), mean)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), mean, atol=1e-12)
    np.testing.assert_array_equal(points, original)
    assert surrogate.fit_result is fit
    np.testing.assert_array_equal(surrogate.predict(points)[1], std)
    assert "confidence" not in " ".join(ax.get_legend_handles_labels()[1])
    points[1, 1] += 0.1
    with pytest.raises(ValueError, match="remain fixed"):
        ra.plotting.plot_surrogate_slice(surrogate, points)


def learning_result():
    history = tuple(
        LearningStep(
            p, 2, n, (p / 2, 2 * p), (1, 3), False, bootstrap_probability_band=band
        )
        for p, n, band in [
            (0.001, 30, (0.0007, 0.0012)),
            (0.002, 35, None),
            (0.0015, 40, (0.0013, 0.0017)),
        ]
    )
    estimate = ReliabilityEstimate(
        0.0016, 0.05, None, None, 1000, "importance_sampling", "independent"
    )
    return ActiveLearningResult(estimate, False, "max_iterations", 40, history)


def test_history_preserves_exploratory_measure_gaps_and_failure_status():
    result = learning_result()
    _, ax = ra.plotting.plot_learning_history(
        result, band="bootstrap", reference_probability=0.00155
    )
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [30, 35, 40])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.001, 0.002, 0.0015])
    assert "max_iterations" in ax.get_title()
    # Missing middle ranges must not be filled by interpolation or final Pf.
    assert len(ax.collections[0].get_paths()) == 2
    assert result.history[1].bootstrap_probability_band is None
    missing = replace(result, history=(result.history[1],))
    with pytest.raises(ValueError, match="No bootstrap"):
        ra.plotting.plot_learning_history(missing, band="bootstrap")


def test_pce_selection_retains_zero_and_reports_unidentifiable_candidates():
    fit = fitted_pce().fit_result
    candidates = (
        replace(fit.candidates[0], degree=1, corrected_loo_error=0.2),
        replace(fit.candidates[0], degree=2, corrected_loo_error=0),
        replace(fit.candidates[0], degree=3, corrected_loo_error=np.inf),
    )
    _, ax = ra.plotting.plot_pce_selection(replace(fit, candidates=candidates))
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.2, 0])
    assert ax.get_yscale() == "symlog"
    assert "1 nonfinite" in ax.texts[0].get_text()


def test_strong_maximum_uses_completed_sample_without_new_draws():
    form, _ = normal_form()
    analysis = ra.StrongMaximumTest(form, point_number=40, rng=71)
    with pytest.raises(ValueError, match="completed two-dimensional"):
        ra.plotting.plot_strong_maximum(analysis)
    result = analysis.run()
    points = result.points_u.copy()
    count = analysis._evaluation_count
    _, ax = ra.plotting.plot_strong_maximum(result)
    plotted = np.concatenate(
        [collection.get_offsets() for collection in ax.collections[:4]]
    )
    np.testing.assert_allclose(np.sort(plotted, axis=0), np.sort(points, axis=0))
    np.testing.assert_array_equal(result.points_u, points)
    assert analysis._evaluation_count == count


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": ((1, -1), (0, 1))},
        {"bounds": ((0, 1), (0, np.inf))},
        {"n_points": True},
        {"batch_size": 0},
        {"labels": ("x",)},
        {"density_levels": [0.1]},
        {"density": lambda points: -np.ones(len(points))},
    ],
)
def test_invalid_limit_state_plot_inputs(kwargs):
    arguments = {"bounds": ((-1, 1), (-1, 1)), "n_points": 5, **kwargs}
    with pytest.raises(ValueError):
        ra.plotting.plot_limit_state(lambda points: points[:, 0], **arguments)


def test_limit_state_rejects_ambiguous_batch_output():
    with pytest.raises(ValueError, match="one finite value per row"):
        ra.plotting.plot_limit_state(
            lambda points: points.T, bounds=((-1, 1), (-1, 1)), n_points=5
        )
