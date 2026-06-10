# Checkpoints

| file | what it is |
|------|------------|
| `D32K16.pt` | supervised pre-training (SFT) of the D32 K16 dualGNN |
| `reinforce.pt` | same model after REINFORCE fine-tuning toward uniform-over-pool |

These ship as package data, so they are available from any install (pip
wheel included). The REINFORCE-finetuned checkpoint is the default for
inference (better uniformity over the FRT pool); the SFT-only checkpoint
is included for comparison.

```python
from dualgnn.model import DualGNN
net = DualGNN.default()        # the shipped reinforce.pt, cached per device

# or load either file explicitly:
from importlib import resources
net = DualGNN.from_ckpt(resources.files("dualgnn") / "ckpts" / "D32K16.pt")
```
