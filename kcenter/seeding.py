"""
First-point (seeding) strategies — research focus #1.

The 2x guarantee holds for ANY first point, so in theory the choice is free.
On a specific dataset the achieved radius is not free at all, and that gap is
what these strategies probe.

    strategy                 cost of choosing the seed      needs mesh?
    -----------------------  ----------------------------   -----------
    random                   O(1)                           no
    centroid-nearest         O(n)                           no
    farthest-from-centroid   O(n)                           no
    densest-kdtree           O(n log n)                     no
    mst-skeleton-dense       O(n^2)   <- the naive version   no
    delaunay-mst-center      O(n log n) <- same answer       yes
    delaunay-density         O(n)  (mesh already built)      yes
    largest-empty-circle     O(n log n)                      yes

The two Delaunay rows are the interesting ones: `delaunay-mst-center` returns
exactly the same point as `mst-skeleton-dense` but drops the cost from O(n^2)
to O(n log n), because the Euclidean MST is a subgraph of the Delaunay
triangulation.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from . import geometry as geo


# --------------------------------------------------------------------------- #
# Strategies that need no triangulation
# --------------------------------------------------------------------------- #


def seed_random(X, rng, mesh=None) -> int:
    """Textbook default. O(1). Varies run to run."""
    return int(rng.integers(len(X)))


def seed_centroid(X, rng, mesh=None) -> int:
    """Point nearest the centre of mass. O(n). Spread outward from the middle."""
    return int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())


def seed_extreme(X, rng, mesh=None) -> int:
    """Point farthest from the centre of mass. O(n). The deliberate opposite."""
    return int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmax())


def seed_densest_kdtree(X, rng, mesh=None, knn: int = 10) -> int:
    """Densest point by mean distance to k nearest neighbours. O(n log n)."""
    knn = min(knn, len(X) - 1)
    d, _ = cKDTree(X).query(X, k=knn + 1)
    return int(d[:, 1:].mean(axis=1).argmin())


def seed_mst_center_dense(X, rng, mesh=None) -> int:
    """
    MST skeleton centre the naive way: dense distance matrix, MST, then exact
    eccentricity by all-pairs shortest path. O(n^2) memory AND time.

    Kept in the codebase purely as the baseline that the Delaunay version below
    is measured against — same output, much worse complexity.
    """
    D = cdist(X, X)
    mst = minimum_spanning_tree(D)
    sp = shortest_path(mst, directed=False)
    return int(sp.max(axis=1).argmin())


# --------------------------------------------------------------------------- #
# Strategies built on the Delaunay mesh
# --------------------------------------------------------------------------- #


def seed_delaunay_mst_center(X, rng, mesh=None) -> int:
    """
    Same MST-skeleton idea, computed properly:
        Delaunay          O(n log n)
        Kruskal on ~3n edges  O(n log n)
        double-sweep tree centre  O(n)
    Total O(n log n) instead of O(n^2).
    """
    mesh = mesh if mesh is not None else geo.build_mesh(X)
    edges, weights = geo.emst(mesh)
    return geo.tree_center(mesh.n, edges, weights)


def seed_delaunay_density(X, rng, mesh=None) -> int:
    """
    Densest point measured by mean incident Delaunay edge length. O(n) once the
    mesh exists — no separate KD-tree needed, which is the point of reusing one
    structure for everything.
    """
    mesh = mesh if mesh is not None else geo.build_mesh(X)
    return int(geo.neighbor_stats(mesh)["mean"].argmin())


def seed_largest_empty_circle(X, rng, mesh=None) -> int:
    """
    Start at the data point nearest the largest empty circle of the whole set —
    the biggest hole in the data. The circle's centre is a Delaunay circumcentre,
    so this is O(n log n).

    Rationale: Gonzalez spends its budget covering gaps, so open the process at
    the largest gap instead of discovering it three centers later.
    """
    mesh = mesh if mesh is not None else geo.build_mesh(X)
    cc, radii = geo.circumcenters(X, mesh.simplices)
    ok = np.isfinite(radii)
    if not ok.any():
        return seed_centroid(X, rng)
    best = cc[ok][np.argmax(radii[ok])]
    return int(np.linalg.norm(X - best, axis=1).argmin())


SEEDERS = {
    "random": seed_random,
    "centroid-nearest": seed_centroid,
    "farthest-from-centroid": seed_extreme,
    "densest-kdtree": seed_densest_kdtree,
    "delaunay-density": seed_delaunay_density,
    "delaunay-mst-center": seed_delaunay_mst_center,
    "largest-empty-circle": seed_largest_empty_circle,
}

#: seeders that read the triangulation (used to report mesh-build cost fairly)
MESH_SEEDERS = {"delaunay-density", "delaunay-mst-center", "largest-empty-circle"}
