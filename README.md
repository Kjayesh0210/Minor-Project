# k-Center with Delaunay Triangulation — Project 3

A compact implementation of the geometric k-center problem using a single Delaunay
triangulation to speed up center selection, outlier filtering, and exact coverage
candidates for the greedy farthest-point algorithm.

This project is designed to be read in two layers:

- a quick-start workflow for running the experiments locally, and
- a technical summary of the research questions and results.

## Quick start

### Prerequisites

- Python 3.9+
- pip

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the experiment suite

```bash
python run_demo.py
```

This executes the six experiments and writes results to `results/`.

To run a smaller custom study:

```bash
python run_demo.py --k 8 --n 600 --instances 20
python run_demo.py --k 3 --n 60 --instances 1 --seed 0
```

### 3) View the interactive browser demo

Open `web/index.html` in any browser. It is fully self-contained and does not need a
local server or internet access.

### 4) Rebuild the PDF write-up

```bash
python make_pdf.py
```

This regenerates the project explanation PDF.

---

## What this project studies

Core: **Gonzalez's greedy farthest-point algorithm**, O(nk), 2× guarantee.
Geometric spine: **one Delaunay triangulation**, built once in O(n log n), reused for
first-center selection, outlier detection, and the Voronoi coverage regions.

Two research focuses, as agreed:

1. **Which point should be the first center?**
2. **How should outliers be removed?**

---

## Project structure

| Path | Contents |
|---|---|
| `kcenter/geometry.py` | Delaunay mesh, EMST from Delaunay edges, tree centre, circumcentres, largest empty circle |
| `kcenter/core.py` | Gonzalez, multi-start, Hochbaum–Shmoys, k-means, local search, brute-force optimum |
| `kcenter/seeding.py` | Seven first-point strategies, three of them Delaunay-derived |
| `kcenter/outliers.py` | Delaunay detector, robust Gonzalez, Delaunay filter, Charikar greedy |
| `kcenter/datasets.py` | Five dataset families, each with a ground-truth outlier mask |
| `kcenter/viz.py` | Figures: mesh, Voronoi regions, study charts |
| `run_demo.py` | The six experiments |
| `make_pdf.py` | Builds the explanation PDF from the results |
| `web/index.html` | Interactive simulator, self-contained, works offline |
| `results/` | Generated figures, CSV, full console output |
| `Project3_kCenter_Delaunay_Explanation.pdf` | The write-up for the next review |

---

## The central idea

One triangulation, three jobs, each of which would otherwise cost O(n²):

| Job | Without Delaunay | With Delaunay | Measured |
|---|---|---|---|
| MST-skeleton seeding | O(n²) dense MST + all-pairs | O(n log n) | **10.3× faster, same answer** |
| Outlier detection | O(n²) or a separate KD-tree | O(n) on the existing mesh | **F1 0.89 vs 0.63** |
| Farthest point on a continuous map | O(res²·k) grid scan | O(k log k), exact interior | **619× fewer candidates** |

Two facts make this work, and both are worth quoting in the review:

- The **Euclidean MST is a subgraph of the Delaunay triangulation** (Shamos & Hoey, 1975),
  so Kruskal only needs the ~3n Delaunay edges rather than all n(n−1)/2 pairs.
- The **Voronoi diagram is the dual of the Delaunay triangulation**, so every Voronoi
  vertex is a triangle circumcentre — the largest-empty-circle candidate set is finite
  and exact, not sampled.

---

## Findings

**Focus 1 — first center.** The starting point swings the achieved radius by **25–40%**
within a single instance. But across 12 instances × 5 dataset families, *no cheap
deterministic rule reliably beats a random start* — the best win on only 2 of 5 families.
Multi-start ×25 is best on 5 of 5. The one clear geometric win is non-convex data: on
two-moons the Delaunay MST centre is the best deterministic rule.

This is a **negative result, honestly measured**, and it is more defensible than a
cherry-picked positive one. The 2× bound is start-independent, so there was never reason
to expect a universal best seed.

**Focus 2 — outliers.** The design decision that mattered: scoring a point by the
**mean** length of its incident Delaunay edges works badly (F1 0.63), because points on
the *rim* of a cluster have long edges reaching across the gap. Scoring by the
**minimum** incident edge length gives F1 **0.89** — and that minimum *is* the exact
nearest-neighbour distance, because the Delaunay graph contains the nearest-neighbour
graph. O(n), no KD-tree.

Also: abandoning 20 of 400 points (5%) cuts the radius by **20%**.

**Ratio to the true optimum** (brute force, n=45, k=3): Gonzalez averages **1.54×**,
worst observed 1.89×. With local-search polish, **1.05×**. The "2×" is a ceiling, not a
typical result.

**Measured complexity.** Dense MST seeding fits n^2.25; the Delaunay version fits n^1.02.
The O(n²) → O(n log n) claim is confirmed by measurement, not asserted.

---

## Known limitations — fix these before the final report

1. **The detector threshold was tuned on the data it was scored on.** The MAD threshold
   of 8.0 came from the same sweep it is evaluated against. Pick it on one set of
   instances and report on a different set.
2. **All data is synthetic**, and planted outliers are uniform random — the easiest case
   for a density detector. One real dataset would change the weight of the whole project.
3. **The seeding conclusion is a negative result and needs more power.** 12 instances
   shows the differences are smaller than the spread; a paired Wilcoxon test over ~30
   instances would make the claim properly.
4. **Multi-start is not cost-matched** — it uses 25× the compute, so of course it wins.
   Compare at equal time budget.
5. **The browser triangulation is O(n²)**, not O(n log n): Bowyer–Watson with naive point
   location, chosen for readability. The Python side uses Qhull.
6. **Border candidates in the largest-empty-circle step are still sampled** (100 per
   edge). Interior candidates are exact.
7. **The Delaunay filter loses on radius** against Charikar's method. Its case is that it
   needs no z parameter and is 30× faster — not that it is more accurate.

---

## Suggested next steps, in order of value per hour

1. **Write up the local-search swap** — already implemented, and it is the strongest
   empirical result in the project (1.05× optimal vs 1.54× plain).
2. **Exact Voronoi/border intersections** — an afternoon, removes the last approximation.
3. **Capacity limits** — the most realistic constraint for warehouses; the theory bound
   (9×) is loose enough that a simple heuristic can look good.
4. **Incremental placement** — connects to the 2026 incrementality result in the research
   note.
5. **A real dataset** — city or delivery coordinates.
