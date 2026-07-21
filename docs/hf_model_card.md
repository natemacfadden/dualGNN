---
license: gpl-3.0
pipeline_tag: graph-ml
library_name: dualgnn
tags:
- gnn
- sampling
- triangulations
- lattice-polytopes
- calabi-yau
- string-theory
---

# dualGNN

[![PyPI](https://img.shields.io/pypi/v/dualgnn)](https://pypi.org/project/dualgnn/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/natemacfadden/dualGNN/blob/main/tutorials/inference_demo.ipynb)

dualGNN is an autoregressive message-passing GNN for sampling fine, regular
triangulations (FRTs) of convex lattice polytopes. It operates on a
generalization of the dual graph of a triangulation, with edges labeled by
"signed circuits" -- combinatorial invariants from oriented matroid theory
that are provably necessary and empirically sufficient for exposing regularity.
The model is independent of the number of points in the polytope, invariant
under its orientation-preserving symmetries, and guarantees (in 2D) that every
rollout is a fine triangulation. On unseen polygons it is the most uniform FRT
sampler we tested, with ~92k parameters, trained in ~7.5 hours on a single
consumer GPU. Applied to string theory, it uniformly samples Calabi-Yau
threefolds at $h^{1,1} = 86$ (consistent with uniformity at
$h^{1,1} = 128$) -- an order of magnitude beyond previous learned methods
with a ~1300x smaller model.

The model was presented in the paper [Sampling Triangulations and Calabi-Yau Threefolds with Autoregressive GNNs](https://arxiv.org/abs/2605.27770).

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

### String Theory Application

Pair the 2D sampler with the [NTFE algorithm](https://arxiv.org/abs/2309.10855) to sample fine, regular, star triangulations (FRSTs) of a reflexive 4D polytope:

```python
import numpy as np
from cytools import Polytope
from dualgnn.model import DualGNN
from dualgnn.ntfe  import sample_ntfes

verts = [[-1, -1, -1, -1], [-1, -1, -1,  3], [-1, -1,  3, -1], [-1,  3, -1, -1],
         [ 1, -1, -1, -1], [ 1, -1, -1,  3], [ 1, -1,  3, -1], [ 1,  3, -1, -1]]
poly = Polytope(np.array(verts, dtype=np.int64))                           # reflexive, h11 = 86
net  = DualGNN.default()
heights = sample_ntfes(poly, net, N=20, N_face_triangs=1_000, n_workers=4) # (20, npts) float64
```

## Limitations

- The fineness guarantee holds in 2D; regularity is not guaranteed per rollout
  (`only_regular=True` filters by rejection).
- K=16 message-passing rounds cap the effective graph diameter; very large
  polygons may exceed it.
- Uniformity is validated to h^{1,1}=86; at h^{1,1}=128 diagnostics are
  consistent with uniformity but weaker.
- Paper figures are not reproducible from the shipped inference code alone
  (see the repo README).

## Links

- **Paper:** [Sampling Triangulations and Calabi-Yau Threefolds with Autoregressive GNNs](https://arxiv.org/abs/2605.27770) (arXiv:2605.27770)
- **Code / training scripts:** [github.com/natemacfadden/dualGNN](https://github.com/natemacfadden/dualGNN)
- **Interactive Demo:** [Open in Colab](https://colab.research.google.com/github/natemacfadden/dualGNN/blob/main/tutorials/inference_demo.ipynb)
- **Benchmark protocol:** [`eval/`](https://github.com/natemacfadden/dualGNN/tree/main/eval)
- **Archive:** [doi:10.5281/zenodo.20622920](https://doi.org/10.5281/zenodo.20622920)

## Citation

```bibtex
@article{MacFadden:2605.27770,
  author        = {MacFadden, Nate},
  title         = {Sampling Triangulations and Calabi-{Y}au Threefolds with Autoregressive {GNN}s},
  year          = {2026},
  eprint        = {2605.27770},
  archivePrefix = {arXiv},
  primaryClass  = {hep-th},
  doi           = {10.48550/arXiv.2605.27770},
  url           = {https://arxiv.org/abs/2605.27770},
}
```
