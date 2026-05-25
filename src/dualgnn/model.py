# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  dualGNN model: per-node logit for next-simp prediction.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


# main model class
# ================
class _Layer(nn.Module):
    """
    One round of message passing on dualGNN's graph.

    Each forward pass:
        1) build a per-edge message `(circ, LayerNorm(f))` from each sender
        2) scatter messages to receivers under reductions `agg = (sum,min,max)`
        3) MLP on `(LayerNorm(f), agg, metadata)` for
           `metadata = (placed, legal)`
        4) residual add: `f_new = f + delta`

    Note: `agg` summarizes messages from a node's neighbors, so it does NOT
    include its own feature vector. Step 4 therefore passes its own `f`
    separately, alongside `agg` and `metadata`, to preserve self-state across
    layers.

    The per-node `metadata` (placed, legal) is re-injected at every layer
    rather than only at init, so every layer can directly see the current
    placement/legality state.

    Parameters
    ----------
    D : int
        Hidden state dimension.
    """

    def __init__(self, D):
        super().__init__()

        # dimensions
        self.D         = D
        self.Dmetadata = 2 # (placed, legal)
        self.Dedge     = 4 # for 2D polytopes

        msg_dim = self.Dedge + D # circuit + feature vector
        agg_dim = 3 * msg_dim    # (sum, min, max)

        # layer pieces
        self.norm = nn.LayerNorm(D)
        self.mlp  = nn.Sequential(
            nn.Linear(D + agg_dim + self.Dmetadata, D), # +D to also pass self f
            nn.GELU(),
            nn.Linear(D, D),
        )

    def forward(self, f, circ_features, edge_indices, metadata):
        """
        Apply one message-passing layer.

        Parameters
        ----------
        f : Tensor
            `(batch_size, Nsimps, D)` float per-node hidden state.
        circ_features : Tensor
            `(Nedges, Dedge)` float per-edge circuit features (sender's
            perspective), shared across the batch.
        edge_indices : Tensor
            `(2, Nedges)` long. Row 0 = sender simp index, row 1 = receiver.
        metadata : Tensor
            `(batch_size, Nsimps, Dmetadata)` float per-node state features
            (placed, legal).

        Returns
        -------
        f_new : Tensor
            `(batch_size, Nsimps, D)` float updated hidden state.
        """
        # read dimensions/shapes
        batch_size, Nsimps, D = f.shape
        Nedges  = circ_features.shape[0]
        msg_dim = self.Dedge + D

        # message routing
        src     = edge_indices[0]
        dst     = edge_indices[1]

        # form the messages
        # -----------------
        f_norm   = self.norm(f)
        f_sender = f_norm[:, src, :]

        # add the circ_features to each feature vector
        msg = torch.cat([
            circ_features.unsqueeze(0).expand(batch_size, -1, -1),
            f_sender,
        ], dim=-1)

        # aggregation containers
        agg_sum = torch.zeros(batch_size, Nsimps, msg_dim,
                              device=f.device, dtype=msg.dtype)
        agg_min = torch.zeros(batch_size, Nsimps, msg_dim,
                              device=f.device, dtype=msg.dtype)
        agg_max = torch.zeros(batch_size, Nsimps, msg_dim,
                              device=f.device, dtype=msg.dtype)

        # send the messages
        idx     = dst.view(1, Nedges, 1).expand(batch_size, Nedges, msg_dim)
        agg_sum.scatter_reduce_(1, idx, msg, reduce="sum",  include_self=False)
        agg_min.scatter_reduce_(1, idx, msg, reduce="amin", include_self=False)
        agg_max.scatter_reduce_(1, idx, msg, reduce="amax", include_self=False)

        agg = torch.cat([agg_sum, agg_min, agg_max], dim=-1)
        return f + self.mlp(torch.cat([f_norm, agg, metadata], dim=-1))

class DualGNN(nn.Module):
    """
    The dualGNN model: AR sampler of simplices for uniform sampling of
    fine regular triangulations. Analogous to a Pointer Network.

    Stack of `K` `_Layer`s with a shared (per-polygon) dual-graph structure.

    Output is one logit per candidate simp; downstream code applies log-softmax
    over the candidate axis to get a distribution over which simp to place next.

    Parameters
    ----------
    D : int, optional
        Hidden state dimension. Default 32.
    K : int, optional
        Number of `_Layer` rounds. Default 16.
    """

    def __init__(self, *, D=32, K=16):
        super().__init__()
        # hyperparameters
        self.D = D
        self.K = K

        # unchanging config
        self.Dedge     = 4 # (for 2D polytopes)
        self.Dmetadata = 2 # (placed, legal)

        # for setting initial feature vector data... one hot
        self.init_mlp = nn.Sequential(
            nn.Linear(self.Dmetadata, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # message passing/simplex selection data
        self.layers = nn.ModuleList(
            [_Layer(D) for _ in range(K)]
        )
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)

    @classmethod
    def from_ckpt(
        cls,
        path:   str | Path,
        device: str | torch.device | None = None,
    ) -> "DualGNN":
        """
        Load a `DualGNN` from a checkpoint produced by `Trainer` or
        `reinforce`. Reads `D` and `K` from the ckpt's hparams, strips the
        `_orig_mod.` prefix left by `torch.compile`, and returns the model
        in eval mode on `device`.

        Parameters
        ----------
        path : str or Path
            Checkpoint file.
        device : str or torch.device, optional
            Target device. Default `None` -> CUDA if available, else CPU.

        Returns
        -------
        net : DualGNN
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        hp   = ckpt["hparams"]
        net  = cls(D=hp["d_model"], K=hp["k_rounds"]).to(device).eval()
        net.load_state_dict({
            k.replace("_orig_mod.", ""): v
            for k, v in ckpt["state_dict"].items()
        })
        return net

    def forward(self, circ_features, edge_indices, placed, legal):
        """
        Score the next simp to place for each batch element. Does so via
            1) initializing hidden states via `init_mlp(metadata)`,
            2) running `K` `_Layer` message passing rounds,
            3) projecting to a single logit per simp, and then
            4) masking placed / illegal entries to `-inf`.

        Parameters
        ----------
        circ_features : Tensor
            `(Nedges, Dedge)` float circuit features for each edge.
        edge_indices : Tensor
            `(2, Nedges)` long. Row 0 = sender simp, row 1 = receiver.
        placed : Tensor
            `(batch_size, Nsimps)` bool or float. `1` where simp `i` is
            in the current partial triangulation of batch element `b`.
        legal : Tensor
            `(batch_size, Nsimps)` bool or float. `1` where simp `i` is
            unplaced AND pairwise-compatible with every placed simp in
            row `b` (see `sampler.compute_legal`).

        Returns
        -------
        logits : Tensor
            `(batch_size, Nsimps)` float. `-inf` for placed or non-legal
            entries. Apply `log_softmax(dim=-1)` for the predicted distribution
            over which simp to place next.
        """
        metadata = torch.cat([
            placed.float().unsqueeze(-1),
            legal.float().unsqueeze(-1),
        ], dim=-1)

        f = self.init_mlp(metadata)

        # K message-passing rounds
        for layer in self.layers:
            f = layer(f, circ_features, edge_indices, metadata)

        # project to logits
        f      = self.norm(f)
        logits = self.head(f).squeeze(-1)

        # mask and return
        mask = placed.bool() | (~legal.bool())
        return logits.masked_fill(mask, float("-inf"))
