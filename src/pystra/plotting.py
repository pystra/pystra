"""Reusable reliability figures; plotting never starts or refits an analysis."""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .active_learning import ActiveLearningResult, PCEFitResult, Surrogate
from .results import FORMResult
from .strong_maximum import StrongMaximumTest

__all__ = [
    "plot_limit_state",
    "plot_form_geometry",
    "plot_surrogate_slice",
    "plot_learning_history",
    "plot_pce_selection",
    "plot_strong_maximum",
]


def _axes(ax):
    if ax is None:
        import matplotlib.pyplot as plt

        return plt.subplots(figsize=(6, 4), layout="constrained")
    return ax.figure, ax


def _integer(value, name, minimum):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _points(points, dimension=None):
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or len(points) < 2
        or points.shape[1] < 1
        or not np.all(np.isfinite(points))
        or (dimension is not None and points.shape[1] != dimension)
    ):
        raise ValueError(
            "points must be a finite (n_points, n_variables) array, n_points >= 2"
        )
    return points


def _evaluate(function, points, batch_size):
    values = []
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        value = np.asarray(function(batch.copy()), dtype=float)
        if value.shape != (len(batch),) or not np.all(np.isfinite(value)):
            raise ValueError("The batch callable must return one finite value per row")
        values.append(value)
    return np.concatenate(values)


def _prediction(surrogate, points, batch_size):
    means, spreads = [], []
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        mean, std = (
            np.asarray(value, dtype=float) for value in surrogate.predict(batch.copy())
        )
        if (
            mean.shape != (len(batch),)
            or std.shape != mean.shape
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(std))
            or np.any(std < 0)
        ):
            raise ValueError(
                "Surrogate predictions must be finite vectors with nonnegative std"
            )
        means.append(mean)
        spreads.append(std)
    return np.concatenate(means), np.concatenate(spreads)


def plot_limit_state(
    limit_state: Callable,
    *,
    bounds: Sequence[Sequence[float]],
    surrogate: Optional[Surrogate] = None,
    density: Optional[Callable] = None,
    density_levels: Optional[Sequence[float]] = None,
    n_points: int = 151,
    batch_size: int = 2048,
    labels: Sequence[str] = ("$x_1$", "$x_2$"),
    shade_failure: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot a two-dimensional limit-state boundary and optional surrogate.

    Parameters
    ----------
    limit_state : callable
        Receives row-wise points of shape (n, 2); returns shape (n,).
        Failure is g <= 0. The plot evaluates n_points**2 true-model points;
        this is additional plotting cost, outside analysis evaluation counts.
    bounds : array-like, shape (2, 2)
        Increasing (lower, upper) bounds for each coordinate, in its own units.
    surrogate : Surrogate, optional
        Fitted surrogate in exactly the same coordinates as limit_state.
        PySTRA active-learning surrogates require independent normal coordinates;
        map the true model accordingly before supplying it here.
    density : callable, optional
        Joint density in the plotted coordinates, with the same batch contract.
    density_levels : sequence of float, optional
        Positive increasing density contours. Automatic levels when omitted.
    n_points : int, default 151
        Grid points per coordinate. A grid may miss small failure regions;
        the figure is not a probability estimate or validation certificate.
    batch_size : int, default 2048
        Maximum rows per model, density or surrogate call.
    labels : pair of str
        Axis labels; include physical units when applicable.
    shade_failure : bool, default True
        Shade the safe and failure regions of the true model.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.

    Returns
    -------
    figure, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes, without calling show or changing analysis state.
    """
    n_points = _integer(n_points, "n_points", 2)
    batch_size = _integer(batch_size, "batch_size", 1)
    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.shape != (2, 2)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("bounds must contain two finite increasing pairs")
    if len(labels) != 2:
        raise ValueError("labels must contain two coordinate labels")
    if density_levels is not None:
        levels = np.asarray(density_levels, dtype=float)
        if (
            density is None
            or levels.ndim != 1
            or not len(levels)
            or not np.all(np.isfinite(levels))
            or np.any(levels <= 0)
            or np.any(np.diff(levels) <= 0)
        ):
            raise ValueError(
                "density_levels requires a density and positive increasing levels"
            )
    grid = np.meshgrid(*(np.linspace(low, high, n_points) for low, high in bounds))
    points = np.column_stack([coordinate.ravel() for coordinate in grid])
    values = _evaluate(limit_state, points, batch_size).reshape(grid[0].shape)
    predicted = (
        None
        if surrogate is None
        else _prediction(surrogate, points, batch_size)[0].reshape(values.shape)
    )
    densities = (
        None
        if density is None
        else _evaluate(density, points, batch_size).reshape(values.shape)
    )
    if densities is not None and np.any(densities < 0):
        raise ValueError("density must be nonnegative")
    fig, ax = _axes(ax)
    if shade_failure:
        ax.contourf(
            *grid, values <= 0, levels=[-0.5, 0.5, 1.5], colors=["#edf4f8", "#f1c6b8"]
        )
    for field, color, style, label in (
        (values, "black", "-", "Limit-state boundary"),
        (predicted, "tab:orange", "--", "Surrogate boundary"),
    ):
        if field is not None and field.min() < 0 < field.max():
            ax.contour(*grid, field, levels=[0], colors=color, linestyles=style)
            ax.plot([], [], color=color, linestyle=style, label=label)
    if densities is not None:
        options = {} if density_levels is None else {"levels": density_levels}
        ax.contour(*grid, densities, colors="navy", linewidths=0.7, **options)
    ax.set(
        xlabel=labels[0],
        ylabel=labels[1],
        xlim=bounds[0],
        ylim=bounds[1],
        aspect="equal",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


def plot_form_geometry(
    result: FORMResult,
    *,
    boundary: Optional[np.ndarray] = None,
    tangent_length: float = 4.0,
    label: str = "FORM",
    color: str = "C0",
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot a converged two-dimensional FORM point, radius and tangent.

    Parameters
    ----------
    result : FORMResult
        Converged snapshot with two reference coordinates.
    boundary : ndarray, shape (n, 2), optional
        Ordered boundary points already transformed into the result's reference
        coordinates and variable order. No transformation is inferred or run.
    tangent_length : float, default 4
        Half-length of the displayed tangent in reference-space units.
    label, color : str
        Legend prefix and color, useful for overlaying transformation orders.
    ax : matplotlib.axes.Axes, optional
        Existing axes. This plot performs no model evaluations.

    Returns
    -------
    figure, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes. Nonconverged or higher-dimensional results are rejected;
        a projection would not preserve the displayed design-point geometry.
    """
    point = np.asarray(result.standard_point, dtype=float).ravel()
    alpha = np.asarray(result.alpha, dtype=float).ravel()
    if (
        not result.converged
        or point.shape != (2,)
        or alpha.shape != (2,)
        or not np.all(np.isfinite([point, alpha]))
        or np.linalg.norm(alpha) == 0
    ):
        raise ValueError("A converged two-dimensional FORM result is required")
    if not np.isfinite(tangent_length) or tangent_length <= 0:
        raise ValueError("tangent_length must be finite and positive")
    if boundary is not None:
        boundary = _points(boundary, 2)
    direction = np.array([alpha[1], -alpha[0]]) / np.linalg.norm(alpha)
    tangent = point + np.array([-tangent_length, tangent_length])[:, None] * direction
    fig, ax = _axes(ax)
    if boundary is not None:
        ax.plot(*boundary.T, color=color, label=f"{label} boundary")
    ax.plot(*tangent.T, "--", color=color, label=f"{label} tangent")
    ax.plot(*point, "o", color=color, label=f"{label} design point")
    ax.plot([0, point[0]], [0, point[1]], ":", color=color, alpha=0.6)
    ax.plot(0, 0, "+", color="black")
    coordinate = "u" if result.standard_space == "normal" else "v"
    ax.set(xlabel=f"${coordinate}_1$", ylabel=f"${coordinate}_2$", aspect="equal")
    ax.legend()
    return fig, ax


def plot_surrogate_slice(
    surrogate: Surrogate,
    points: np.ndarray,
    *,
    coordinate: int = 0,
    limit_state: Optional[Callable] = None,
    n_std: float = 2.0,
    batch_size: int = 2048,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot a fitted surrogate along one independent-normal coordinate.

    Parameters
    ----------
    surrogate : Surrogate
        Fitted surrogate; prediction only, without training or RNG draws.
    points : ndarray, shape (n, n_variables)
        Slice points. The selected coordinate must strictly increase; all other
        coordinates must remain fixed. Rows are independent-normal coordinates.
    coordinate : int, default 0
        Varying coordinate column.
    limit_state : callable, optional
        True response with batch input (n, n_variables) and output (n,).
        Must use the same coordinates; these n evaluations are additional cost.
    n_std : float, default 2
        Surrogate-spread multiplier, not a confidence level. Bootstrap spread
        and Kriging predictive std have different meanings; neither establishes
        true-model coverage or includes all model-selection uncertainty.
    batch_size : int, default 2048
        Maximum rows in each prediction/evaluation call.
    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    figure, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes, without displaying them.
    """
    points = _points(points)
    coordinate = _integer(coordinate, "coordinate", 0)
    batch_size = _integer(batch_size, "batch_size", 1)
    if coordinate >= points.shape[1] or np.any(np.diff(points[:, coordinate]) <= 0):
        raise ValueError("coordinate must select a strictly increasing column")
    fixed = np.delete(points, coordinate, axis=1)
    if np.any(fixed != fixed[0]):
        raise ValueError("All other coordinates must remain fixed along a slice")
    if not np.isfinite(n_std) or n_std < 0:
        raise ValueError("n_std must be finite and nonnegative")
    mean, std = _prediction(surrogate, points, batch_size)
    truth = None if limit_state is None else _evaluate(limit_state, points, batch_size)
    fig, ax = _axes(ax)
    locations = points[:, coordinate]
    if truth is not None:
        ax.plot(locations, truth, color="black", label="True limit state")
    ax.plot(locations, mean, "--", color="tab:blue", label="Surrogate mean")
    ax.fill_between(
        locations,
        mean - n_std * std,
        mean + n_std * std,
        alpha=0.25,
        color="tab:blue",
        label=f"Surrogate spread (±{n_std:g} std)",
    )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(
        xlabel=f"$u_{coordinate + 1}$ (other coordinates fixed)", ylabel="Limit state"
    )
    ax.legend()
    return fig, ax


def plot_learning_history(
    result: ActiveLearningResult,
    *,
    band: Optional[str] = "probability",
    reference_probability: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot exploratory Pf against cumulative true-model evaluations.

    Parameters
    ----------
    result : ActiveLearningResult
        Snapshot with a nonempty history. The title retains its stopping status.
        Historical estimates use each exploration estimator's sampling measure;
        the independent final estimate is not substituted into this curve.
    band : {'probability', 'bootstrap', None}, default 'probability'
        Plot stored mean±2std probability sensitivity or actual bootstrap Pf
        ranges. Neither is a confidence interval or a bound on true Pf.
        Missing bootstrap ranges leave gaps; entirely absent ranges raise.
    reference_probability : float, optional
        Independently obtained probability in [0, 1], drawn for comparison.
    ax : matplotlib.axes.Axes, optional
        Existing axes. No estimation or model evaluation is performed.

    Returns
    -------
    figure, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes, without displaying them.
    """
    if not result.history:
        raise ValueError("result must contain a learning history")
    if band not in ("probability", "bootstrap", None):
        raise ValueError("band must be 'probability', 'bootstrap', or None")
    if reference_probability is not None and (
        not np.isfinite(reference_probability) or not 0 <= reference_probability <= 1
    ):
        raise ValueError("reference_probability must be finite and in [0, 1]")
    calls = np.array([step.n_evaluations for step in result.history])
    probabilities = np.array([step.failure_probability for step in result.history])
    ranges = None
    if band is not None:
        field = (
            "probability_band"
            if band == "probability"
            else "bootstrap_probability_band"
        )
        ranges = np.array(
            [
                (
                    getattr(step, field)
                    if getattr(step, field) is not None
                    else (np.nan, np.nan)
                )
                for step in result.history
            ]
        )
        if not np.any(np.isfinite(ranges)):
            raise ValueError(f"No {band} ranges were recorded")
    fig, ax = _axes(ax)
    if ranges is not None:
        label = (
            "Mean ±2 std probability sensitivity"
            if band == "probability"
            else "Bootstrap Pf range"
        )
        ax.fill_between(calls, ranges[:, 0], ranges[:, 1], alpha=0.25, label=label)
    ax.plot(calls, probabilities, "o-", label="Exploratory failure probability")
    if reference_probability is not None:
        ax.axhline(
            reference_probability,
            color="black",
            linestyle="--",
            label="Reference probability",
        )
    ax.set(
        xlabel="True model evaluations",
        ylabel="$P_f$",
        title=f"Active learning: {result.status}",
    )
    ax.legend()
    return fig, ax


def plot_pce_selection(
    fit: PCEFitResult, *, ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot corrected LOO scores by candidate degree and q-norm.

    Parameters
    ----------
    fit : PCEFitResult
        Fitted PCE diagnostics. Scores condition on selected supports; they are
        not reliability errors. Nonfinite candidates are omitted and counted
        on the figure. Exact zero scores use a symmetric-log axis.
    ax : matplotlib.axes.Axes, optional
        Existing axes. No model selection is rerun.

    Returns
    -------
    figure, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes, without displaying them.
    """
    if fit is None or not fit.candidates:
        raise ValueError("A fitted PCE with candidate diagnostics is required")
    valid = [c for c in fit.candidates if np.isfinite(c.corrected_loo_error)]
    if not valid:
        raise ValueError("No finite candidate scores to plot")
    fig, ax = _axes(ax)
    markers = ("o", "s", "^", "D", "v", "P")
    for index, q in enumerate(sorted({c.q_norm for c in valid})):
        candidates = sorted((c for c in valid if c.q_norm == q), key=lambda c: c.degree)
        ax.plot(
            [c.degree for c in candidates],
            [c.corrected_loo_error for c in candidates],
            linestyle="-",
            marker=markers[index % len(markers)],
            label=f"q = {q:g}",
        )
    errors = np.array([c.corrected_loo_error for c in valid])
    if np.all(errors > 0):
        ax.set_yscale("log")
    else:
        positive = errors[errors > 0]
        ax.set_yscale(
            "symlog", linthresh=float(positive.min() / 10) if len(positive) else 1e-12
        )
    omitted = len(fit.candidates) - len(valid)
    if omitted:
        ax.text(
            0.02,
            0.98,
            f"{omitted} nonfinite candidate score(s) omitted",
            transform=ax.transAxes,
            va="top",
        )
    ax.set(
        xlabel="Candidate degree",
        ylabel="Corrected LOO error",
        title="Sparse PCE model selection",
    )
    ax.legend()
    return fig, ax


def plot_strong_maximum(
    analysis: StrongMaximumTest, *, ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot two-dimensional Strong Maximum Test sample groups in normal space.

    Parameters
    ----------
    analysis : StrongMaximumTest
        Successfully completed two-dimensional diagnostic. Its stored points,
        candidate and test sphere are drawn without evaluations or RNG draws.
        The status is shown; no competing region detected is not a certificate
        of FORM accuracy. Higher-dimensional projections are rejected.
    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    figure, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes, without displaying them.
    """
    from matplotlib.patches import Circle

    if not analysis.results_valid or analysis.nrv != 2:
        raise ValueError("A completed two-dimensional Strong Maximum Test is required")
    fig, ax = _axes(ax)
    colors = {
        "near_failure": "tab:blue",
        "far_failure": "tab:red",
        "near_safe": "tab:orange",
        "far_safe": "0.65",
    }
    markers = {
        "near_failure": "o",
        "far_failure": "x",
        "near_safe": "^",
        "far_safe": "+",
    }
    for group, color in colors.items():
        points = analysis.get_points(group)
        ax.scatter(
            *points.T,
            s=12,
            color=color,
            marker=markers[group],
            label=f"{group.replace('_', ' ')} ({len(points)})",
        )
    ax.add_patch(
        Circle((0, 0), analysis.radius, fill=False, linestyle=":", color="grey")
    )
    ax.scatter(
        *analysis.design_point, marker="*", s=150, color="black", label="Candidate"
    )
    ax.set(
        xlabel="$u_1$",
        ylabel="$u_2$",
        title=analysis.status.replace("_", " "),
        aspect="equal",
    )
    ax.legend()
    return fig, ax
