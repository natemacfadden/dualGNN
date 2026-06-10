---
license: gpl-3.0
pipeline_tag: graph-ml
tags:
- gnn
- sampling
- triangulations
- lattice-polytopes
- calabi-yau
- string-theory
---

# dualGNN

dualGNN is an autoregressive message-passing GNN for sampling fine, regular
triangulations (FRTs) of convex lattice polytopes. It operates on a
generalization of the dual graph of a triangulation, with edges labeled by
"signed circuits" -- combinatorial invariants from oriented matroid theory
that are necessary and sufficient for exposing regularity. The model is
independent of the number of points in the polytope, invariant under its
orientation-preserving symmetries, and guarantees (in 2D) that every rollout
is a fine triangulation. On unseen polygons it is the most uniform FRT
sampler tested, with ~92k parameters, trained in ~7.5 hours on a single
consumer GPU. Applied to string theory, it uniformly samples Calabi-Yau
threefolds at $h^{1,1} = 86$ (consistent with uniformity at
$h^{1,1} = 128$) -- an order of magnitude beyond previous learned methods
with a ~1000x smaller model.

## Files

| file | what it is |
|------|------------|
| `reinforce.pt` | D=32, K=16 model after REINFORCE fine-tuning -- **the default** |
| `D32K16.pt` | the same model before fine-tuning (SFT only), for comparison |

## Usage

The weights here are the same files bundled inside the
[`dualgnn`](https://pypi.org/project/dualgnn/) pip package, so the simplest
path needs nothing from this page:

```python
# pip install dualgnn
import numpy as np
from dualgnn import sample_frts

pts = np.array([[x, y] for x in range(5) for y in range(5)])  # [0,4]^2
fts = sample_frts(pts, 1000, only_regular=True, seed=0)
```

To load this repo's checkpoint explicitly:

```python
from huggingface_hub import hf_hub_download
from dualgnn.model import DualGNN

path = hf_hub_download("natemacfadden/dualGNN", "reinforce.pt")
net  = DualGNN.from_ckpt(path)
```

## Links

- **Paper:** [Sampling Triangulations and Calabi-Yau Threefolds with
  Autoregressive GNNs](https://arxiv.org/abs/2605.27770) (arXiv:2605.27770)
- **Code / training scripts:**
  [github.com/natemacfadden/dualGNN](https://github.com/natemacfadden/dualGNN)
- **Benchmark protocol** (evaluate your own sampler against dualGNN):
  [`eval/`](https://github.com/natemacfadden/dualGNN/tree/main/eval)
- **Archive:** [doi:10.5281/zenodo.20622920](https://doi.org/10.5281/zenodo.20622920)

## Citation

```bibtex
@article{MacFadden:2605.27770,
  author  = {MacFadden, Nate},
  title   = {Sampling Triangulations and Calabi-{Y}au Threefolds with Autoregressive {GNN}s},
  doi     = {10.48550/arXiv.2605.27770},
  url     = {https://arxiv.org/abs/2605.27770},
}
```
