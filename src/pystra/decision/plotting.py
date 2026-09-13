"""Presentation of design decision study results."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .objectives import annualized_safety_cost

__all__ = ["plot_summary"]


def _default_decision_plot_quantities(data: pd.DataFrame) -> list[str]:
    candidates = [
        "pf",
        "annual_failure_rate",
        "annualized_safety_cost",
        "acceptability_margin",
        "lqi_marginal_term",
        "screening_margin",
        "objective",
    ]
    return [column for column in candidates if column in data]


def _coerce_reference_designs(reference_designs) -> list[dict[str, Any]]:
    if reference_designs is None:
        return []

    if isinstance(reference_designs, Mapping):
        if "design" in reference_designs:
            return [dict(reference_designs)]
        return [
            {"label": str(label), "design": design}
            for label, design in reference_designs.items()
        ]

    if np.isscalar(reference_designs):
        return [{"design": float(reference_designs)}]

    references = []
    for item in reference_designs:
        if isinstance(item, Mapping):
            references.append(dict(item))
        else:
            references.append({"design": float(item)})
    return references


def plot_summary(
    data: Any,
    design: str,
    quantities: Optional[Sequence[str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    yscales: Optional[Mapping[str, str]] = None,
    reference_designs: Optional[Any] = None,
    target_failure_probability: Optional[float] = None,
    target_label: str = "LQI target",
    invert_yaxis: Optional[Iterable[str]] = None,
    panel_labels: bool = False,
    axes: Optional[Sequence[Any]] = None,
    figsize: Optional[tuple[float, float]] = None,
    line_kwargs: Optional[Mapping[str, Any]] = None,
):
    """Plot a one-dimensional design decision optimization summary.

    The helper is intentionally generic: it expects a dataframe-like object
    with a design column and one or more result columns.  It is suitable for
    continuous one-dimensional design sweeps.  More complex discrete
    alternatives or scenario studies can still use the same risk quantities,
    but usually need a problem-specific visualization.

    Parameters
    ----------
    data : dataframe-like
        Table containing the design values and result quantities to plot.
    design : str
        Name of the design-variable column.
    quantities : sequence of str, optional
        Result columns to plot.  When omitted, common decision columns such as
        ``pf``, ``annualized_safety_cost``, ``screening_margin``, and ``objective``
        are used when present.
    labels : mapping, optional
        Axis label overrides keyed by column name.  The design column may also
        be included to set the shared x-axis label.
    yscales : mapping, optional
        Matplotlib y-scale overrides keyed by quantity column name.  Failure
        probability columns use ``"log"`` by default.
    reference_designs : sequence or mapping, optional
        Designs to mark with vertical lines and point markers.  Each item may
        be a scalar design value or a mapping with ``design``, ``label``,
        ``color``, and ``marker`` keys.  A mapping without a ``design`` key is
        interpreted as ``{label: design}``.
    target_failure_probability : float, optional
        Drawn as a horizontal line on failure-probability panels.
    target_label : str, optional
        Legend label for ``target_failure_probability``.
    invert_yaxis : iterable of str, optional
        Quantity columns whose y-axis should be inverted.
    panel_labels : bool, optional
        If ``True``, label panels ``A)``, ``B)``, ...
    axes : sequence, optional
        Existing matplotlib axes.  Its length must match the number of
        quantities.
    figsize : tuple, optional
        Figure size used when ``axes`` is not supplied.
    line_kwargs : mapping, optional
        Keyword arguments passed to the main line plots.

    Returns
    -------
    tuple
        ``(fig, axes)`` from matplotlib.
    """

    df = pd.DataFrame(data).copy()
    if design not in df:
        raise KeyError(f"Design column {design!r} is missing")
    if df.empty:
        raise ValueError("data must contain at least one design row")

    plot_quantities = (
        list(quantities)
        if quantities is not None
        else _default_decision_plot_quantities(df)
    )
    if not plot_quantities:
        raise ValueError("No decision quantities available to plot")

    missing = [column for column in plot_quantities if column not in df]
    if missing:
        raise KeyError(f"Plot quantities are missing: {', '.join(missing)}")

    labels = {} if labels is None else dict(labels)
    yscales = {} if yscales is None else dict(yscales)
    invert = set() if invert_yaxis is None else set(invert_yaxis)
    references = _coerce_reference_designs(reference_designs)

    df = df.sort_values(design)
    x = df[design].astype(float).to_numpy()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("plot_summary requires matplotlib") from exc

    if axes is None:
        if figsize is None:
            figsize = (7.5, max(2.5, 2.0 * len(plot_quantities)))
        fig, axes = plt.subplots(len(plot_quantities), 1, figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(axes)
        if len(axes) != len(plot_quantities):
            raise ValueError("axes length must match quantities length")
        fig = axes[0].figure

    main_line = {"color": "#27349a", "marker": "o", "linewidth": 1.8}
    if line_kwargs is not None:
        main_line.update(line_kwargs)

    reference_defaults = [
        {"color": "#2f3aa6", "marker": "o"},
        {"color": "#d62728", "marker": "s"},
        {"color": "#2ca02c", "marker": "^"},
        {"color": "#9467bd", "marker": "D"},
    ]

    for index, (axis, quantity) in enumerate(zip(axes, plot_quantities)):
        y = df[quantity].astype(float).to_numpy()
        axis.plot(x, y, **main_line)
        axis.set_ylabel(labels.get(quantity, quantity))
        axis.grid(True, alpha=0.3)

        yscale = yscales.get(quantity)
        if yscale is None and quantity in {"pf", "annual_failure_rate"}:
            yscale = "log"
        if yscale is not None:
            axis.set_yscale(yscale)
        if quantity in invert:
            axis.invert_yaxis()

        if panel_labels:
            axis.text(
                0.02,
                0.86,
                f"{chr(ord('A') + index)})",
                transform=axis.transAxes,
                fontsize="large",
                fontstyle="italic",
            )

        if quantity in {
            "screening_margin",
            "acceptability_margin",
            "lqi_marginal_term",
        }:
            axis.axhline(0.0, color="0.35", linewidth=0.9)

        if target_failure_probability is not None and quantity in {
            "pf",
            "annual_failure_rate",
        }:
            axis.axhline(
                target_failure_probability,
                color="#d62728",
                linestyle="--",
                linewidth=1.0,
                label=target_label,
            )

        for reference_index, reference in enumerate(references):
            if "design" not in reference:
                raise KeyError("Each reference design mapping requires a 'design' key")
            style = reference_defaults[reference_index % len(reference_defaults)]
            color = reference.get("color", style["color"])
            marker = reference.get("marker", style["marker"])
            label = reference.get("label")
            design_value = float(reference["design"])
            line_label = label if index == 0 and label else None
            axis.axvline(
                design_value,
                color=color,
                linewidth=1.0,
                alpha=0.75,
                label=line_label,
            )
            if x[0] <= design_value <= x[-1]:
                y_value = float(np.interp(design_value, x, y))
                axis.plot(
                    design_value,
                    y_value,
                    marker=marker,
                    markersize=8,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    linestyle="none",
                )

        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, fontsize="small", loc="best")

    axes[-1].set_xlabel(labels.get(design, design))
    fig.tight_layout()
    return fig, axes
