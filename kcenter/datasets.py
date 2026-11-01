"""
Synthetic datasets.

Every generator returns (X, truth) where `truth` is a boolean mask marking
points that were PLANTED as outliers. Having ground truth is what lets
experiment 3 measure detection precision and recall instead of just eyeballing
the picture.

Swap in real coordinates by returning (your_array, np.zeros(n, bool)).
"""

from __future__ import annotations

import numpy as np


def uniform(n=400, seed=0):
    """Flat random spread — structureless, the hardest case for any seeding rule."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n, 2)), np.zeros(n, dtype=bool)


def blobs(n=400, clusters=8, spread=4.0, seed=0):
    """Clustered 'towns' — what delivery and warehouse data actually looks like."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(10, 90, size=(clusters, 2))
    sizes = np.full(clusters, n // clusters)
    sizes[: n - sizes.sum()] += 1
    X = np.vstack([c + rng.normal(0, spread, size=(s, 2))
                   for c, s in zip(centers, sizes)])
    return X, np.zeros(len(X), dtype=bool)


def blobs_with_outliers(n=400, clusters=8, spread=4.0, n_out=8, seed=0):
    """Clusters plus planted strays — the case the whole outlier section is about."""
    rng = np.random.default_rng(seed + 99)
    X, _ = blobs(n - n_out, clusters, spread, seed)
    out = rng.uniform(-60, 160, size=(n_out, 2))
    X = np.vstack([X, out])
    truth = np.zeros(len(X), dtype=bool)
    truth[-n_out:] = True
    return X, truth


def two_moons(n=400, noise=2.0, seed=0):
    """Curved, non-convex clusters — where a centroid falls in empty space."""
    rng = np.random.default_rng(seed)
    m = n // 2
    t1 = np.linspace(0, np.pi, m)
    t2 = np.linspace(0, np.pi, n - m)
    a = np.column_stack([50 * np.cos(t1), 50 * np.sin(t1)])
    b = np.column_stack([50 * np.cos(t2) + 50, -50 * np.sin(t2) + 25])
    X = np.vstack([a, b]) + rng.normal(0, noise, size=(n, 2))
    return X, np.zeros(n, dtype=bool)


def grid_with_hole(n=400, seed=0):
    """
    A near-uniform field with a large empty region punched out of it — built to
    make the largest-empty-circle seeding rule earn or lose its keep.
    """
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(0, 100, size=2)
        if np.linalg.norm(p - np.array([62, 40])) < 22:  # the hole
            continue
        pts.append(p)
    X = np.array(pts)
    return X, np.zeros(len(X), dtype=bool)


DATASETS = {
    "uniform": uniform,
    "clustered": blobs,
    "clustered+outliers": blobs_with_outliers,
    "two-moons": two_moons,
    "field+hole": grid_with_hole,
}
