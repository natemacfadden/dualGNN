# dualGNN
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

A small graph-neural-network sampler for fine regular triangulations (FRTs) of 2D lattice polygons. Trains the next-simp conditional `P(sigma | T_partial)` on a harvested FRT pool (supervised), then fine-tunes toward uniform-over-pool with REINFORCE.

**Scope:** convex 2D lattice polygons only. In theory you could manually construct a graph corresponding to a non-convex polygon and it'd work, but that is very OOD.

## Install

```
conda env create -f environment.yml
conda activate dualgnn
```

(Or `pip install -e .` if you already have a compatible torch + CYTools environment.)

## Inference

Load a trained checkpoint and sample FRTs of a polygon:

```python
import numpy as np
from dualgnn       import DualGraph, sample
from dualgnn.model import DualGNN

pts = np.array([[x, y] for x in range(5) for y in range(5)], dtype=np.int64)  # [0,4]^2
net = DualGNN.from_ckpt("ckpts/reinforce.pt")
fts = sample(net, DualGraph(pts), Ntriangs=8)          # (8, 32, 3) int8
```

See `notebooks/inference_demo.ipynb` for a runnable version with plotting.

For comparison, two reference samplers are bundled (CYTools-free, both
return `(simps, status)`):

```python
from dualgnn import grow2d, pushing

simps, status = grow2d(pts, seed=0)    # random fine triangulation
simps, status = pushing(pts, seed=0)   # random fine pushing triangulation
```

## Train end-to-end

Three commands -- generate polygons, supervised train, REINFORCE fine-tune:

```
# 1) sample polygons (Npts 5..40, 3 per bucket); writes polygons.parquet
python scripts/make_polygons.py --out runs/data/polygons.parquet

# 2) supervised train (~5 h on a Blackwell-class GPU at 500k steps)
python scripts/train.py \
    --run-dir runs/sft \
    --src-polygons runs/data/polygons.parquet \
    --src-fts-dir  runs/data/fts \
    --n-steps 500000

# 3) REINFORCE fine-tune from the SFT final ckpt (~2 h at 10k steps)
python scripts/reinforce.py \
    --init-ckpt runs/sft/ckpt_0500000.pt \
    --run-path  runs/rl \
    --steps 10000
```

The FRT pool is auto-harvested per polygon on first use; you can also
pre-harvest a specific polygon via `python scripts/harvest.py --poly-id N`.

## Layout

```
src/dualgnn/          library code (DualGraph, DualGNN, sampler, training)
scripts/              CLI entry points (train, reinforce, harvest, make_polygons, visualize)
ckpts/                shipped checkpoints (D32K16 SFT, D32K16 + REINFORCE)
notebooks/            inference demo
```
