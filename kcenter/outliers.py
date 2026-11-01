"""
Outlier handling — research focus #2.

Why plain Gonzalez is fragile: its rule is "take the single farthest point", so
one stray sensor reading or a mistyped coordinate becomes a center. With k = 6
and 8 planted outliers, half the budget can end up on empty ground.

Three responses, with different costs and guarantees:

    method                     complexity        guarantee   needs a guess of z?
    -------------------------  ----------------  ----------  -------------------
    robust_gonzalez            O(nk)             none        yes
    delaunay_filter_gonzalez   O(n log n + nk)   none        no (threshold instead)
    charikar_outliers          O(n^2 k log n)    3x          yes

The Delaunay route is the one this project contributes: it does not ask the user
to guess z. It reads local density straight off the triangulation already built
for seeding and the Voronoi regions, so the marginal cost is O(n).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.distance import cdist

from . import geometry as geo
from .core import Solution, evaluate


# --------------------------------------------------------------------------- #
# Objective when z points are allowed to be abandoned
# --------------------------------------------------------------------------- #


def evaluate_with_outliers(X, center_idx, z: int):
    """
    Radius needed to cover the best n - z points. O(nk).

    Taking the (z+1)-th largest distance rather than the largest is the whole
    definition of k-center-with-outliers.
    """
    center_idx = np.asarray(center_idx)
    D = cdist(X, X[center_idx])
    dmin = D.min(axis=1)
    n = len(X)
    if z <= 0:
        return float(dmin.max()), np.zeros(n, dtype=bool)
    z = min(z, n - 1)
    cut = np.partition(dmin, n - z - 1)[n - z - 1]
    dropped = dmin > cut
    return float(cut), dropped


# --------------------------------------------------------------------------- #
# 1. Detection: local density from the Delaunay mesh
# --------------------------------------------------------------------------- #


def delaunay_outlier_scores(X, mesh=None, stat: str = "min") -> np.ndarray:
    """
    Outlier score per point, normalised by the median absolute deviation.
    O(n) on top of the mesh.

    stat="min" (default): score from the point's nearest-neighbour distance,
    read straight off the mesh because the Delaunay triangulation contains the
    nearest-neighbour graph. A point is an outlier when even its CLOSEST
    neighbour is far away.

    stat="mean": score from the mean incident edge length. This is the obvious
    first thing to try and it works badly on clustered data — every point on the
    rim of a cluster has long edges reaching across the gap to the next cluster,
    so it gets flagged too. Kept here because experiment 3 reports both, and the
    gap between them is a result worth showing rather than hiding.

    MAD rather than standard deviation, because the values we are hunting are
    exactly the ones that would inflate a standard deviation and so hide
    themselves inside their own threshold.
    """
    mesh = mesh if mesh is not None else geo.build_mesh(X)
    raw = geo.neighbor_stats(mesh)[stat]
    med = np.median(raw)
    mad = np.median(np.abs(raw - med))
    scale = 1.4826 * mad
    if scale < 1e-12:
        scale = raw.std() if raw.std() > 1e-12 else 1.0
    return (raw - med) / scale


def flag_outliers(X, threshold: float = 3.5, mesh=None, stat: str = "min"):
    """
    Boolean mask of suspected outliers. O(n log n) including the mesh.

    `threshold` is in robust-sigma units. 3.5 is the textbook MAD cutoff, but
    experiment 3 sweeps it rather than trusting the convention.
    """
    scores = delaunay_outlier_scores(X, mesh, stat)
    return scores > threshold, scores


def detection_quality(flagged: np.ndarray, truth: np.ndarray) -> dict:
    """Precision / recall / F1 of the detector against planted outliers."""
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


# --------------------------------------------------------------------------- #
# 2. Robust Gonzalez: ignore the z farthest when choosing
# --------------------------------------------------------------------------- #


def robust_gonzalez(X, k: int, z: int, first: int = 0) -> Solution:
    """
    Gonzalez, but each step takes the (z+1)-th farthest point instead of the
    farthest. Cost stays O(nk): np.argpartition finds the (z+1)-th largest in
    O(n) without a full sort.

    No approximation guarantee — it is a heuristic — and it needs z up front.
    """
    t0 = time.perf_counter()
    n = len(X)
    z = max(0, min(z, n - k - 1))

    centers = [int(first)]
    dmin = np.linalg.norm(X - X[first], axis=1)

    for _ in range(k - 1):
        if z == 0:
            nxt = int(dmin.argmax())
        else:
            top = np.argpartition(dmin, -(z + 1))[-(z + 1):]
            nxt = int(top[np.argmin(dmin[top])])  # the (z+1)-th largest overall
        centers.append(nxt)
        np.minimum(dmin, np.linalg.norm(X - X[nxt], axis=1), out=dmin)

    centers = np.array(centers)
    seconds = time.perf_counter() - t0
    radius, dropped = evaluate_with_outliers(X, centers, z)
    _, mean_dist, assignment = evaluate(X, centers)
    return Solution(f"Robust Gonzalez (z={z})", centers, radius, mean_dist,
                    assignment, seconds, extra={"dropped": dropped})


# --------------------------------------------------------------------------- #
# 3. Delaunay filter, then ordinary Gonzalez
# --------------------------------------------------------------------------- #


def delaunay_filter_gonzalez(X, k: int, threshold: float = 3.5,
                             seeder=None, rng=None, mesh=None) -> Solution:
    """
    Detect outliers from the triangulation, drop them, run plain Gonzalez on the
    survivors. O(n log n) for the mesh + O(nk) for the greedy.

    The practical advantage over robust_gonzalez: no need to know z in advance.
    The detector decides how many points are anomalous from the geometry itself.
    Reported radius covers the survivors; `extra["radius_all"]` reports what it
    costs to also cover the points that were thrown away, so nothing is hidden.
    """
    t0 = time.perf_counter()
    mesh = mesh if mesh is not None else geo.build_mesh(X)
    flagged, scores = flag_outliers(X, threshold, mesh)

    keep_idx = np.flatnonzero(~flagged)
    if len(keep_idx) < k:  # detector was too aggressive; fall back to everything
        keep_idx = np.arange(len(X))
        flagged = np.zeros(len(X), dtype=bool)

    Xk = X[keep_idx]
    first_local = 0 if seeder is None else seeder(Xk, rng)

    centers_local = [int(first_local)]
    dmin = np.linalg.norm(Xk - Xk[first_local], axis=1)
    for _ in range(k - 1):
        nxt = int(dmin.argmax())
        centers_local.append(nxt)
        np.minimum(dmin, np.linalg.norm(Xk - Xk[nxt], axis=1), out=dmin)

    centers = keep_idx[np.array(centers_local)]
    seconds = time.perf_counter() - t0

    radius_survivors = float(dmin.max())
    radius_all, _, assignment = evaluate(X, centers)
    mean_dist = float(cdist(X, X[centers]).min(axis=1).mean())

    return Solution(
        f"Delaunay filter + Gonzalez (t={threshold})", centers, radius_survivors,
        mean_dist, assignment, seconds,
        extra={"dropped": flagged, "n_dropped": int(flagged.sum()),
               "radius_all": radius_all, "scores": scores},
    )


# --------------------------------------------------------------------------- #
# 4. Charikar et al. style greedy — the one with a proof
# --------------------------------------------------------------------------- #


def charikar_outliers(X, k: int, z: int) -> Solution:
    """
    Greedy ball-covering with a 3x guarantee (Charikar et al. 2001 style).
    O(n^2 k log n): binary search over O(n^2) candidate radii, each test doing k
    passes over an n x n neighbourhood matrix.

    For a candidate r: repeatedly take the point whose radius-r ball covers the
    most still-uncovered points, then mark everything within 3r as covered.
    Feasible if k balls cover at least n - z points.

    Included so the project has one outlier method with an actual proof behind
    it, to sit against the two fast heuristics above.
    """
    t0 = time.perf_counter()
    n = len(X)
    D = cdist(X, X)
    candidates = np.unique(np.round(D[np.triu_indices(n, k=1)], 9))

    def try_radius(r):
        covered = np.zeros(n, dtype=bool)
        within_r = D <= r
        within_3r = D <= 3 * r
        chosen = []
        for _ in range(k):
            gain = (within_r & ~covered).sum(axis=1)
            p = int(gain.argmax())
            if gain[p] == 0:
                break
            chosen.append(p)
            covered |= within_3r[p]
        return chosen if covered.sum() >= n - z else None

    lo, hi, best = 0, len(candidates) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        got = try_radius(candidates[mid])
        if got:
            best, hi = got, mid - 1
        else:
            lo = mid + 1

    centers = np.array(best if best else [0])
    seconds = time.perf_counter() - t0
    radius, dropped = evaluate_with_outliers(X, centers, z)
    _, mean_dist, assignment = evaluate(X, centers)
    return Solution(f"Charikar 3x (z={z})", centers, radius, mean_dist,
                    assignment, seconds, extra={"dropped": dropped})
