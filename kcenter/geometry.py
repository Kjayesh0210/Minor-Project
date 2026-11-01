"""
Delaunay triangulation layer.

This is the structural idea of the whole project: ONE triangulation, built once
in O(n log n), then reused for three different jobs that would otherwise each
cost O(n^2):

    job                        without Delaunay      with Delaunay
    -------------------------  --------------------  ----------------------
    MST-skeleton seeding       O(n^2) dense MST      O(n log n)
    local-density / outliers   O(n^2) or KD-tree     O(n log n), no extra tree
    Voronoi coverage + LEC     O(grid) scan          O(k log k), exact

Why the MST speed-up is legal: the Euclidean minimum spanning tree is a SUBGRAPH
of the Delaunay triangulation (Shamos & Hoey, 1975). So Kruskal only ever needs
to look at the ~3n Delaunay edges, never all n(n-1)/2 pairs.

Why the Voronoi part is exact: the Voronoi diagram is the straight-line dual of
the Delaunay triangulation, and every Voronoi vertex is the circumcentre of some
Delaunay triangle. No sampling, no grid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


# --------------------------------------------------------------------------- #
# Building the triangulation
# --------------------------------------------------------------------------- #


@dataclass
class Mesh:
    """A Delaunay triangulation plus the derived structures we reuse."""

    points: np.ndarray  # (n, 2)
    simplices: np.ndarray  # (m, 3) triangle vertex indices
    edges: np.ndarray  # (e, 2) unique undirected edges, i < j
    edge_len: np.ndarray  # (e,) Euclidean length of each edge
    neighbors: list[list[int]]  # adjacency list over the Delaunay graph
    seconds: float

    @property
    def n(self) -> int:
        return len(self.points)


def build_mesh(X: np.ndarray) -> Mesh:
    """
    Delaunay triangulation via Qhull.  O(n log n) expected in 2-D.

    Everything downstream (seeding, outliers, Voronoi) reads from this one
    object, so the O(n log n) is paid exactly once per dataset.
    """
    t0 = time.perf_counter()
    tri = Delaunay(X)

    # Unique undirected edges from the triangle list. Each triangle gives 3
    # edges; every interior edge appears twice, so we dedupe. O(m) = O(n).
    s = tri.simplices
    raw = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]])
    raw = np.sort(raw, axis=1)
    edges = np.unique(raw, axis=0)
    edge_len = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)

    neighbors: list[list[int]] = [[] for _ in range(len(X))]
    for i, j in edges:
        neighbors[i].append(int(j))
        neighbors[j].append(int(i))

    return Mesh(X, s, edges, edge_len, neighbors, time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Euclidean MST, extracted from the Delaunay edge set
# --------------------------------------------------------------------------- #


class _DSU:
    """Union-find with path compression + union by size. Near O(1) amortised."""

    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def emst(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean minimum spanning tree, Kruskal over Delaunay edges only.

    O(n log n): sorting ~3n edges dominates. The dense-matrix alternative is
    O(n^2) just to build the input, which is the version this replaces.

    Returns (tree_edges (n-1, 2), tree_weights (n-1,)).
    """
    order = np.argsort(mesh.edge_len)
    dsu = _DSU(mesh.n)
    keep = []
    for idx in order:
        i, j = mesh.edges[idx]
        if dsu.union(int(i), int(j)):
            keep.append(idx)
            if len(keep) == mesh.n - 1:
                break
    keep = np.array(keep)
    return mesh.edges[keep], mesh.edge_len[keep]


def tree_center(n: int, tree_edges: np.ndarray, weights: np.ndarray) -> int:
    """
    Weighted centre of a tree: the vertex whose farthest distance along the tree
    is smallest. Found with the classic double-sweep, O(n) total.

    1. sweep from any vertex -> farthest vertex u (an endpoint of the diameter)
    2. sweep from u          -> farthest vertex v, recording parents
    3. walk the u..v path and take the vertex minimising max(dist_u, dist_v)

    Step 1-2 rely on the tree property that the farthest vertex from ANY vertex
    is a diameter endpoint. Doing this exactly with all-pairs shortest paths
    would be O(n^2); this is O(n).
    """
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), w in zip(tree_edges, weights):
        adj[i].append((int(j), float(w)))
        adj[j].append((int(i), float(w)))

    def sweep(src: int):
        dist = np.full(n, np.inf)
        parent = np.full(n, -1, dtype=int)
        dist[src] = 0.0
        stack = [src]
        while stack:  # a tree has no cycles, so DFS is enough - no priority queue
            u = stack.pop()
            for v, w in adj[u]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    stack.append(v)
        return dist, parent

    d0, _ = sweep(0)
    u = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    du, par = sweep(u)
    v = int(np.argmax(np.where(np.isfinite(du), du, -1)))
    dv, _ = sweep(v)

    path, cur = [], v
    while cur != -1:
        path.append(cur)
        cur = int(par[cur])

    path = np.array(path)
    return int(path[np.argmin(np.maximum(du[path], dv[path]))])


# --------------------------------------------------------------------------- #
# Local density straight off the mesh  (feeds both seeding and outlier scoring)
# --------------------------------------------------------------------------- #


def neighbor_stats(mesh: Mesh) -> dict:
    """
    Per-point statistics of the incident Delaunay edges. O(n).

    Returns {"mean": ..., "min": ...}

    `mean` is a local-density measure: small means the point sits in a crowd.

    `min` is the exact distance to the point's nearest neighbour, for free.
    That is not an approximation: the Delaunay triangulation CONTAINS the
    nearest-neighbour graph, so the shortest edge at a vertex is genuinely that
    vertex's 1-NN distance. No KD-tree needed.

    The distinction matters for outlier detection. A point on the rim of a
    cluster has some long incident edges reaching across a gap, so its MEAN is
    large and it gets falsely flagged. But it still has a close neighbour inside
    the cluster, so its MIN stays small. A genuinely isolated point has no close
    neighbour at all, so its MIN is large. Using MIN is what separates "on the
    edge of a cluster" from "not in any cluster".
    """
    n = mesh.n
    total = np.zeros(n)
    count = np.zeros(n)
    smallest = np.full(n, np.inf)

    np.add.at(total, mesh.edges[:, 0], mesh.edge_len)
    np.add.at(total, mesh.edges[:, 1], mesh.edge_len)
    np.add.at(count, mesh.edges[:, 0], 1)
    np.add.at(count, mesh.edges[:, 1], 1)
    np.minimum.at(smallest, mesh.edges[:, 0], mesh.edge_len)
    np.minimum.at(smallest, mesh.edges[:, 1], mesh.edge_len)

    count[count == 0] = 1
    smallest[~np.isfinite(smallest)] = 0.0
    return {"mean": total / count, "min": smallest}


# --------------------------------------------------------------------------- #
# Voronoi side of the duality
# --------------------------------------------------------------------------- #


def circumcenters(P: np.ndarray, simplices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Circumcentre and circumradius of every triangle, vectorised. O(m).

    These circumcentres ARE the vertices of the Voronoi diagram — that is the
    Delaunay/Voronoi duality, used directly rather than calling a separate
    Voronoi routine.
    """
    a, b, c = P[simplices[:, 0]], P[simplices[:, 1]], P[simplices[:, 2]]
    ax, ay = a[:, 0], a[:, 1]
    bx, by = b[:, 0], b[:, 1]
    cx, cy = c[:, 0], c[:, 1]

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = np.where(np.abs(d) < 1e-12, np.nan, d)  # degenerate (collinear) triangles

    a2 = ax**2 + ay**2
    b2 = bx**2 + by**2
    c2 = cx**2 + cy**2
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d

    centers = np.column_stack([ux, uy])
    radii = np.linalg.norm(centers - a, axis=1)
    return centers, radii


def largest_empty_circle(center_xy: np.ndarray, bbox, boundary_samples: int = 100):
    """
    The point of the map farthest from every centre — the centre of the largest
    empty circle, i.e. exactly what Gonzalez's next-centre rule wants if centres
    could be placed anywhere rather than only on data points.

    Theorem: that point is a Voronoi vertex of the current centres, or it lies on
    the boundary of the map. Voronoi vertices = Delaunay circumcentres, so:

        O(k log k) to triangulate the k centres
        O(k)       to score their circumcentres

    versus O(res^2) for a grid scan. The interior answer here is EXACT.

    The boundary candidates are still sampled (the honest caveat); computing the
    Voronoi-edge/border intersections exactly would remove that approximation.
    """
    xmin, xmax, ymin, ymax = bbox
    verts = np.empty((0, 2))

    if len(center_xy) >= 3:
        tri = Delaunay(center_xy)
        cc, _ = circumcenters(center_xy, tri.simplices)
        cc = cc[np.isfinite(cc).all(axis=1)]
        inside = (
            (cc[:, 0] >= xmin) & (cc[:, 0] <= xmax)
            & (cc[:, 1] >= ymin) & (cc[:, 1] <= ymax)
        )
        verts = cc[inside]

    t = np.linspace(0, 1, boundary_samples)
    border = np.vstack([
        np.column_stack([xmin + t * (xmax - xmin), np.full_like(t, ymin)]),
        np.column_stack([xmin + t * (xmax - xmin), np.full_like(t, ymax)]),
        np.column_stack([np.full_like(t, xmin), ymin + t * (ymax - ymin)]),
        np.column_stack([np.full_like(t, xmax), ymin + t * (ymax - ymin)]),
    ])

    def best_of(P):
        if len(P) == 0:
            return None, -np.inf
        d = cdist(P, center_xy).min(axis=1)
        i = int(d.argmax())
        return P[i], float(d[i])

    v_xy, v_r = best_of(verts)
    b_xy, b_r = best_of(border)
    from_vertex = v_r >= b_r

    return {
        "xy": v_xy if from_vertex else b_xy,
        "radius": v_r if from_vertex else b_r,
        "from": "voronoi vertex (exact)" if from_vertex else "map boundary (sampled)",
        "vertices": verts,
        "n_vertices": len(verts),
        "n_boundary": len(border),
        "n_candidates": len(verts) + len(border),
        "vertex_radius": v_r,
        "boundary_radius": b_r,
    }


def farthest_point_by_grid(center_xy: np.ndarray, bbox, res: int = 500):
    """Brute-force grid scan, O(res^2 * k). Only used to check the exact method."""
    xmin, xmax, ymin, ymax = bbox
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    d = cdist(G, center_xy).min(axis=1)
    i = int(d.argmax())
    return G[i], float(d[i]), len(G)
