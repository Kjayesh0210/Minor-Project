"""
Figures.

The coverage regions ARE the Voronoi diagram of the chosen centers: each pixel
is coloured by its nearest center. Computing that on a grid keeps the regions
clipped to the map and works for k = 2 or 3, where a true Voronoi diagram has
unbounded cells that matplotlib cannot fill.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist

PALETTE = plt.get_cmap("tab20")
INK = "#1B2A2F"
SIGNAL = "#C3352B"  # reserved for the bottleneck radius
AMBER = "#D98A1F"  # reserved for outliers
GRAPHITE = "#7D8B8A"


def _bbox(X, pad=0.08):
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    dx, dy = (xmax - xmin) * pad, (ymax - ymin) * pad
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def plot_mesh(ax, X, mesh, color=GRAPHITE, lw=0.35, alpha=0.55):
    """Draw the Delaunay edges as hairlines under everything else."""
    segs = np.stack([X[mesh.edges[:, 0]], X[mesh.edges[:, 1]]], axis=1)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                     alpha=alpha, zorder=1))


def plot_solution(ax, X, sol, show_circles=True, res=400, mesh=None,
                  outliers=None, title=None):
    """Voronoi regions, points, centers, coverage circles, optional mesh."""
    xmin, xmax, ymin, ymax = _bbox(X)
    k = len(sol.centers)

    gx, gy = np.meshgrid(np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    region = cdist(grid, X[sol.centers]).argmin(axis=1).reshape(res, res)
    ax.imshow(region, extent=(xmin, xmax, ymin, ymax), origin="lower",
              cmap=PALETTE, alpha=0.20, vmin=0, vmax=19,
              interpolation="nearest", aspect="auto")
    ax.contour(gx, gy, region, levels=np.arange(k) + 0.5,
               colors="0.35", linewidths=0.6)

    if mesh is not None:
        plot_mesh(ax, X, mesh)

    normal = np.ones(len(X), dtype=bool) if outliers is None else ~outliers
    ax.scatter(X[normal, 0], X[normal, 1],
               c=[PALETTE(i % 20) for i in sol.assignment[normal]],
               s=9, edgecolors="none", alpha=0.85, zorder=2)
    if outliers is not None and outliers.any():
        ax.scatter(X[outliers, 0], X[outliers, 1], s=46, marker="^",
                   facecolors="none", edgecolors=AMBER, linewidths=1.5, zorder=3)

    if show_circles:
        for c in sol.centers:
            ax.add_patch(Circle(X[c], sol.radius, fill=False, ec=SIGNAL,
                                lw=0.9, ls="--", alpha=0.75, zorder=4))

    ax.scatter(X[sol.centers, 0], X[sol.centers, 1], marker="*", s=250,
               c=INK, edgecolors="white", linewidths=1.1, zorder=5)

    ax.set_title(title or f"{sol.label}\nradius = {sol.radius:.3f}"
                          f"   ({sol.seconds*1e3:.1f} ms)", fontsize=9.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_panel(X, sols, path, suptitle="", mesh=None, outliers=None):
    n = len(sols)
    cols = 2 if n == 4 else min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, sol in zip(axes, sols):
        plot_solution(ax, X, sol, mesh=mesh, outliers=outliers)
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seed_study(stats, path, title="First-point strategy: mean radius over instances"):
    """
    stats: {dataset: {strategy: (mean, std, n_instances)}}
    Bars with standard-deviation whiskers — repeated instances, so the spread is
    visible and a single lucky run cannot masquerade as a finding.
    """
    datasets = list(stats.keys())
    strategies = list(next(iter(stats.values())).keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.6),
                             sharey=False)
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, datasets):
        means = np.array([stats[ds][s][0] for s in strategies])
        stds = np.array([stats[ds][s][1] for s in strategies])
        best = means.min()
        colors = ["#2a9d8f" if abs(m - best) < 1e-9 else "#8d99ae" for m in means]
        y = np.arange(len(strategies))
        ax.barh(y, means, xerr=stds, color=colors, error_kw={"lw": 1, "ecolor": INK})
        ax.set_yticks(y)
        ax.set_yticklabels(strategies, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("mean max-radius (lower is better)", fontsize=8.5)
        ax.set_title(ds, fontsize=10.5)
        for yi, m in zip(y, means):
            ax.text(m, yi, f" {m:.2f}", va="center", fontsize=8)
        ax.margins(x=0.22)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_outlier_study(zs, curves, detect, path):
    """Left: radius as z grows. Right: detector precision/recall vs threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for label, ys in curves.items():
        ax1.plot(zs, ys, marker="o", ms=4, lw=1.6, label=label)
    ax1.set_xlabel("z  (points allowed to be left uncovered)")
    ax1.set_ylabel("achieved max-radius")
    ax1.set_title("Cost of covering the last few points", fontsize=11)
    ax1.grid(alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=8.5)

    th = detect["thresholds"]
    ax2.plot(th, detect["precision"], marker="o", ms=4, lw=1.6, label="precision")
    ax2.plot(th, detect["recall"], marker="s", ms=4, lw=1.6, label="recall")
    ax2.plot(th, detect["f1"], marker="^", ms=4, lw=1.8, color=SIGNAL, label="F1")
    ax2.set_xlabel("MAD threshold (robust sigmas)")
    ax2.set_ylabel("score")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title("Delaunay outlier detector vs planted truth", fontsize=11)
    ax2.grid(alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scaling(ns, timings, path, fits=None):
    """Measured runtime vs n on log-log axes, with fitted slopes."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label, ys in timings.items():
        slope = f"  (fitted n^{fits[label]:.2f})" if fits and label in fits else ""
        ax.plot(ns, ys, marker="o", ms=4, lw=1.6, label=label + slope)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n  (number of points)")
    ax.set_ylabel("runtime (seconds)")
    ax.set_title("Measured scaling — does the code match the stated complexity?",
                 fontsize=11.5)
    ax.grid(alpha=0.25, which="both")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_voronoi_candidates(X, sol, res, path, mesh=None):
    """The Delaunay-dual shortcut: circumcentres as the only interior candidates."""
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    plot_solution(ax, X, sol, show_circles=False, mesh=mesh,
                  title="Next center via Delaunay circumcentres (largest empty circle)")
    v = res["vertices"]
    if len(v):
        ax.scatter(v[:, 0], v[:, 1], marker="x", s=38, c=AMBER, linewidths=1.3,
                   zorder=6, label=f"Voronoi vertices ({len(v)} exact candidates)")
    ax.add_patch(Circle(res["xy"], res["radius"], fill=False, ec=SIGNAL, lw=1.9,
                        zorder=7))
    ax.scatter(*res["xy"], marker="o", s=95, c=SIGNAL, edgecolors="white",
               zorder=8, label="largest empty circle")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
