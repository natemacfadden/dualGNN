# Checkpoints

| file | what it is |
|------|------------|
| `D32K16.pt` | supervised pre-training (SFT) of the D32 K16 dualGNN |
| `reinforce.pt` | same model after REINFORCE fine-tuning toward uniform-over-pool |

Both checkpoints are loadable via:

```python
from dualgnn.model import DualGNN
net = DualGNN.from_ckpt("ckpts/reinforce.pt")  # or D32K16.pt
```

The REINFORCE-finetuned checkpoint is the default for inference (better
uniformity over the FRT pool); the SFT-only checkpoint is included for
comparison.
