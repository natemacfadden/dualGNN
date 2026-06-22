A lightweight, CPU-runnable check that roughly reproduces the paper's uniformity diagnostics for the `dualGNN` sampler vs the bundled classical baselines (`grow2d`, `pushing`):

- #unique and #collisions among regular draws vs the uniform `N(1 - (1 - 1/N)^M)` prediction, and
- KL to a flat distribution.

## Running

```sh
python eval/uniformity/4x6tri.py
python eval/uniformity/4x4sq.py
```

`4x6tri` is the triangle `conv{(0,0),(0,4),(6,0)}` (N = 405,706); `4x4sq` is the square `[0,4]^2` (N = 735,430,548).

Each script prints the fairness table (`reg.frac`, `#unique`, `#collisions`, `KL`, `sec/draw` for `dualGNN`/`grow2d`/`pushing` vs the `1/N` reference) and caches the sampled FRTs to `samples_<tag>.npz`. **These outputs need no extra dependencies** -- a plain `pip install dualgnn` is enough.

The two figures (`rankfreq_<tag>.png`, `kl_vs_time_<tag>.png`) additionally require matplotlib (`pip install dualgnn[viz]`). Without it, the scripts still print the table and write the cache, and simply skip the figures.

These scripts are meant to be lightweight. This means they **do not** fully recreate paper figures. Additionally, they use hardcoded FRT counts `N` (405,706 for the triangle, 735,430,548 for `[0,4]^2`) rather than enumerating them.

Run with at least a few thousand draws; the script defaults (`4x6tri`, `4x4sq`) are set accordingly. At the default counts the biased samplers (`grow2d`/`pushing`) separate clearly from `dualGNN`, which tracks the uniform reference. With only a few hundred draws, even the biased samplers have not concentrated enough to collide, so every sampler sits at the noise floor and looks identical.

Note `grow2d`/`pushing` are fast but biased. The only other trustworthy-uniform sampler, `flip_walk`, is deliberately omitted: it is not bundled in this repo and is ~4x slower than dualGNN, so it would dominate the runtime of this lightweight demo. Consequently `kl_vs_time` only places dualGNN among the bundled (biased) samplers and cannot, on its own, establish the "most uniform" result. That dualGNN-vs-`flip_walk` comparison is in the paper and in the Results figures of the top-level README (fig. 13).
