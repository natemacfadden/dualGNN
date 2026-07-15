# Evaluating your own sampler against dualGNN

This folder is the paper's benchmark *protocol*, not its data: the data
regenerates quickly, so what a future sampler author needs is the polygon
definitions, the reference counts, and the diagnostics -- all of which are
here.

## What's in it

- **`polygons.json`** -- 22 benchmark polygons:
  - `fig11_00` ... `fig11_19`: the paper's 20 held-out (OOD) polygons,
    $11 \le N_\mathrm{pts} \le 18$, each with its exact FRT count `n_frt`
    and dualGNN's reference irregular-rate at 50k draws.
  - `4x6tri`, `4x4sq`: the enumerable demo polygons from this folder's
    `uniformity/` (405,706 and 735,430,548 FRTs).
- **`evaluate.py`** -- the scoring protocol (library + CLI).

## The protocol

Your sampler draws `M` fine triangulations of a polygon **with
replacement** (do not deduplicate -- collisions are the signal). Draws are
regularity-filtered, then compared to uniform draws from the polygon's
`n_frt` FRTs:

- **#unique / #collisions** vs the uniform expectation
  $N(1 - (1 - 1/N)^M)$, and
- **KL(empirical || uniform)** vs the *noise floor*: the same KL computed
  for actually-uniform draws of the same size. The floor is strictly
  positive at finite `M`; the target is excess KL $\approx 0$, not
  KL $= 0$. With too few draws ($M \ll \sqrt{N}$), nothing collides and
  every sampler sits at the floor -- use the per-polygon `n_frt` to pick
  `M` large enough to discriminate.

Index convention: simplices index rows of the spec's `lattice_points`
list, in that order. Canonicalization (and exact-duplicate detection) is
handled by the evaluator; your `(M, n_simps, 3)` int array just needs
valid row indices.

## Usage

```bash
# list the benchmark polygons
python eval/evaluate.py --polygon list

# score the dualGNN baseline on one OOD polygon
python eval/evaluate.py --polygon fig11_00 --n 20000

# score YOUR sampler: save draws as (M, n_simps, 3) int .npy and run
python eval/evaluate.py --polygon fig11_00 --samples my_draws.npy
```

or from Python:

```python
from eval.evaluate import load_polygons, evaluate
spec    = load_polygons()["fig11_00"]
metrics = evaluate(my_simps, spec)     # dict; see evaluate() docstring
```

A sampler "matches dualGNN" on this benchmark if, at equal `M`, its excess
KL and collision counts track the uniform reference as closely across the
20 OOD polygons (dualGNN's per-polygon reference irregular-rates are in
`polygons.json`; the paper's headline comparison is fig. 13).

## Not included

The paper's flip-distance autocorrelation diagnostic (fig. 14) needs a
flip-graph implementation and is not bundled; see the paper for that
methodology. For a richer runnable demo (rank-frequency plots, per-sampler
throughput (draws/s), and a KL-vs-time speed-vs-uniformity plot against the
bundled `grow2d`/`pushing` baselines), see [`uniformity/`](uniformity/).
