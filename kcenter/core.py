"""
Core k-center algorithms.

Time complexities (n = points, k = centers, m = restarts, I = iterations):

    evaluate                       O(nk)
    gonzalez                       O(nk)          2x guarantee
    gonzalez_multistart            O(mnk)         2x guarantee
    hochbaum_shmoys                O(n^2 log n)   2x guarantee, independent check
    random_centers                 O(nk)          no guarantee
    kmeans_lloyd                   O(nkI)         different objective
    optimal_discrete               O(C(n,k)*nk)   exact, small n only
    local_search_swap              O(t*nk^2)      no guarantee, practical polish
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class Solution:
    """One k-center solution plus the numbers we report."""

    label: str
    centers: np.ndarray
    radius: float  # THE objective: max distance to nearest center
    mean_dist: float  # k-means-style objective, for contrast
    assignment: np.ndarray
    seconds: float
    extra: dict = None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.label}: radius={self.radius:.4f} in {self.seconds*1e3:.2f} ms>"


def evaluate(X: np.ndarray, center_idx) -> tuple[float, float, np.ndarray]:
    """Score a center set. O(nk)."""
    center_idx = np.asarray(center_idx)
    D = cdist(X, X[center_idx])
    nearest = D.argmin(axis=1)
    dmin = D[np.arange(len(X)), nearest]
    return float(dmin.max()), float(dmin.mean()), nearest


# --------------------------------------------------------------------------- #
# Gonzalez's greedy farthest-point traversal
# --------------------------------------------------------------------------- #


def gonzalez(X: np.ndarray, k: int, first: int = 0, label: str = "Gonzalez") -> Solution:
    """
    Greedy farthest-point traversal (Gonzalez 1985). O(nk), 2x guarantee.

    The O(nk) comes from keeping `dmin` = each point's distance to its nearest
    chosen center. Adding a center can only shrink those values, so one
    np.minimum pass per center suffices instead of rescoring all pairs.
    """
    t0 = time.perf_counter()
    n = len(X)
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    centers = [int(first)]
    dmin = np.linalg.norm(X - X[first], axis=1)

    for _ in range(k - 1):
        nxt = int(dmin.argmax())
        centers.append(nxt)
        np.minimum(dmin, np.linalg.norm(X - X[nxt], axis=1), out=dmin)

    centers = np.array(centers)
    seconds = time.perf_counter() - t0
    radius, mean_dist, assignment = evaluate(X, centers)
    return Solution(label, centers, radius, mean_dist, assignment, seconds)


def gonzalez_trace(X: np.ndarray, k: int, first: int = 0) -> list[dict]:
    """
    Same algorithm, but records each step so the web simulator and the report
    can show WHY each center was chosen. O(nk).
    """
    centers = [int(first)]
    dmin = np.linalg.norm(X - X[first], axis=1)
    steps = [{"step": 0, "chosen": int(first), "radius": float(dmin.max()),
              "reason": "first center (seeding rule)"}]
    for s in range(1, k):
        nxt = int(dmin.argmax())
        gap = float(dmin[nxt])
        centers.append(nxt)
        np.minimum(dmin, np.linalg.norm(X - X[nxt], axis=1), out=dmin)
        steps.append({"step": s, "chosen": nxt, "was_at_distance": gap,
                      "radius": float(dmin.max()),
                      "reason": "farthest point from all current centers"})
    return steps


def gonzalez_multistart(X, k, restarts, rng) -> Solution:
    """m independent random starts, keep the smallest radius. O(mnk)."""
    t0 = time.perf_counter()
    best = None
    for _ in range(restarts):
        sol = gonzalez(X, k, int(rng.integers(len(X))))
        if best is None or sol.radius < best.radius:
            best = sol
    best.label = f"Gonzalez multi-start x{restarts}"
    best.seconds = time.perf_counter() - t0
    return best


# --------------------------------------------------------------------------- #
# Hochbaum-Shmoys parametric pruning: an INDEPENDENT 2x algorithm
# --------------------------------------------------------------------------- #


def hochbaum_shmoys(X: np.ndarray, k: int) -> Solution:
    """
    Parametric pruning (Hochbaum & Shmoys 1985). O(n^2 log n), 2x guarantee.

    The optimal radius must be one of the O(n^2) pairwise distances, so binary
    search that sorted list. For a candidate r, greedily pick any uncovered
    point, take it as a center, and delete everything within 2r; if this uses
    <= k centers, r is feasible.

    Slower than Gonzalez but derived completely differently, which makes it a
    genuine cross-check: if both land on the same radius, a bug in one is
    unlikely to be hiding in the other.
    """
    t0 = time.perf_counter()
    n = len(X)
    D = cdist(X, X)
    candidates = np.unique(D[np.triu_indices(n, k=1)])

    def feasible(r: float):
        uncovered = np.ones(n, dtype=bool)
        chosen = []
        while uncovered.any():
            if len(chosen) == k:
                return None
            p = int(np.flatnonzero(uncovered)[0])
            chosen.append(p)
            uncovered &= D[p] > 2 * r
        return chosen

    lo, hi, best = 0, len(candidates) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        got = feasible(candidates[mid])
        if got is not None:
            best, hi = got, mid - 1
        else:
            lo = mid + 1

    centers = np.array(best if best is not None else list(range(min(k, n))))
    seconds = time.perf_counter() - t0
    radius, mean_dist, assignment = evaluate(X, centers)
    return Solution("Hochbaum-Shmoys pruning", centers, radius, mean_dist,
                    assignment, seconds)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def random_centers(X, k, rng) -> Solution:
    """k points at random. O(nk) to score. No guarantee."""
    t0 = time.perf_counter()
    centers = rng.choice(len(X), size=k, replace=False)
    seconds = time.perf_counter() - t0
    radius, mean_dist, assignment = evaluate(X, centers)
    return Solution("Random centers", centers, radius, mean_dist, assignment, seconds)


def kmeans_lloyd(X, k, rng, iters: int = 100) -> Solution:
    """
    Lloyd's algorithm with k-means++ seeding, written out so the project has no
    scikit-learn dependency. O(nkI).

    Minimises MEAN SQUARED distance, not the MAXIMUM — a different objective.
    Centroids are snapped to the nearest real point so the radius is measured on
    the same footing as Gonzalez.
    """
    t0 = time.perf_counter()
    n = len(X)

    C = [X[rng.integers(n)]]
    d2 = ((X - C[0]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        C.append(X[rng.choice(n, p=probs)])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(axis=1))
    C = np.array(C)

    for _ in range(iters):
        lab = cdist(X, C).argmin(axis=1)
        newC = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC

    centers = cdist(C, X).argmin(axis=1)
    seconds = time.perf_counter() - t0
    radius, mean_dist, assignment = evaluate(X, centers)
    return Solution("k-means (Lloyd)", centers, radius, mean_dist, assignment, seconds)


# --------------------------------------------------------------------------- #
# Local search polish  (research direction F in the note)
# --------------------------------------------------------------------------- #


def local_search_swap(X, sol: Solution, max_rounds: int = 20) -> Solution:
    """
    Try replacing each center with a non-center point; keep any swap that lowers
    the radius. Repeat until nothing improves. O(rounds * n * k * n) worst case,
    so use it on a solution you already have rather than from scratch.

    NOTE: this BREAKS the 2x guarantee in the sense that it no longer follows
    the proof - though since it only ever accepts strict improvements over a
    2x-guaranteed start, the result is still <= 2x. It is the k-center analogue
    of 2-opt for TSP: practically useful, theoretically silent.
    """
    t0 = time.perf_counter()
    centers = list(sol.centers)
    best_r = sol.radius
    n = len(X)

    for _ in range(max_rounds):
        improved = False
        for pos in range(len(centers)):
            for cand in range(n):
                if cand in centers:
                    continue
                trial = centers.copy()
                trial[pos] = cand
                r, _, _ = evaluate(X, trial)
                if r < best_r - 1e-12:
                    centers, best_r, improved = trial, r, True
                    break
            if improved:
                break
        if not improved:
            break

    centers = np.array(centers)
    seconds = time.perf_counter() - t0
    radius, mean_dist, assignment = evaluate(X, centers)
    return Solution(f"{sol.label} + local swap", centers, radius, mean_dist,
                    assignment, sol.seconds + seconds)


# --------------------------------------------------------------------------- #
# Exact optimum, for measuring the REAL ratio instead of quoting the bound
# --------------------------------------------------------------------------- #


def optimal_discrete(X, k, max_combos: int = 400_000) -> Solution:
    """Exhaustive search over all k-subsets. O(C(n,k) * nk). Small n only."""
    n = len(X)
    combos = 1
    for i in range(k):
        combos = combos * (n - i) // (i + 1)
    if combos > max_combos:
        raise ValueError(f"{combos:,} combinations is too many; shrink n or k")

    t0 = time.perf_counter()
    D = cdist(X, X)
    best_r, best_c = np.inf, None
    for c in itertools.combinations(range(n), k):
        r = D[list(c)].min(axis=0).max()
        if r < best_r:
            best_r, best_c = r, c
    seconds = time.perf_counter() - t0

    centers = np.array(best_c)
    radius, mean_dist, assignment = evaluate(X, centers)
    return Solution("OPTIMAL (brute force)", centers, radius, mean_dist,
                    assignment, seconds)
