"""Plotting adapters for completed code-calibration studies."""

import numpy as np

__all__ = ["plot_calibration"]


def plot_calibration(
    results, *, target_beta=None, colors=None, ranges=None, ax=None, figsize=(8, 4)
):
    """Plot labelled study envelopes without running analyses or hiding failures.

    Parameters
    ----------
    results : mapping
        Legend labels mapped to CodeCalibrationResult snapshots.
    target_beta : float, optional
        Reference line. It does not change stored study targets or results.
    colors : mapping, optional
        Label-to-Matplotlib-color overrides.
    ranges : sequence of mapping, optional
        Plot annotations with xl, xu, ytext, text, alpha keys, retained from the
        normalized-reliability tutorial. These are presentation settings, not model inputs.
    ax : matplotlib.axes.Axes, optional
        Existing axes. The returned figure is not automatically displayed.
    figsize : tuple, optional
        Size used when creating axes.

    Returns
    -------
    fig, ax
        Figure and axes. Nonconverged studies raise ValueError before plotting.
    """
    import matplotlib.pyplot as plt

    if not results or any(not r.converged for r in results.values()):
        raise ValueError("Plotting requires nonempty, fully converged study results")
    if target_beta is not None and not np.isfinite(target_beta):
        raise ValueError("target_beta must be finite")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    hatches = ("/", "\\", "x", ".", "o", "+")
    for index, (label, result) in enumerate(results.items()):
        x, beta = result.live_load_ratios, result.beta
        color = (colors or {}).get(label)
        # The boundary is the smallest dead-load ratio, independent of grid order.
        (line,) = ax.plot(
            x, beta[np.argmin(result.dead_load_ratios)], ls=":", color=color
        )
        ax.fill_between(
            x,
            beta.min(axis=0),
            beta.max(axis=0),
            alpha=0.5,
            facecolor=line.get_color(),
            edgecolor=line.get_color(),
            hatch=hatches[index % len(hatches)],
            label=label,
        )
    for r in ranges or ():
        ax.axvspan(r["xl"], r["xu"], color="k", alpha=r["alpha"])
        ax.annotate(
            "",
            xy=(r["xl"], r["ytext"]),
            xytext=(r["xu"], r["ytext"]),
            arrowprops={"arrowstyle": "<->"},
        )
        ax.text((r["xl"] + r["xu"]) / 2, 1.01 * r["ytext"], r["text"], ha="center")
    if target_beta is not None:
        ax.axhline(target_beta, color="k", ls="--", label=rf"$\beta_T={target_beta:g}$")
    ax.set_xlabel(r"Variable load ratio, $a_q=Q/(G+P+Q)$")
    ax.set_ylabel(r"Reliability index, $\beta$")
    ax.grid(True, ls=":")
    ax.legend()
    return ax.figure, ax
