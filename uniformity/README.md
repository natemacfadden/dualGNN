A lightweight, CPU-runnable check that roughly reproduces the paper's uniformity diagnostics for the `dualGNN` sampler vs the bundled classical baselines (`grow2d`, `pushing`):

- #unique and #collisions among regular draws vs the uniform `N(1 − (1 − 1/N)^M)` prediction, and
- KL to a flat distribution.

These scripts are meant to be lightweight. This means they **do not** fully recreate paper figures. Additionally, they use hardcoded FRT counts `N` (405,706 for the triangle, 735,430,548 for `[0,4]^2`) rather than enumerating them.

Note `grow2d`/`pushing` are fast but biased; the speed-vs-uniformity (Pareto) comparison against the trustworthy-uniform `flip_walk` sampler is in the paper, not here, so `kl_vs_time` only shows where dualGNN sits among the bundled samplers.
